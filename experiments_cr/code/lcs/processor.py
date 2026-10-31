"""vLLM (v1, 0.30) batch logits processor: truncation samplers + per-step logging.

Requests opt in through SamplingParams.extra_args["lcs"] (or extra_args["lcs_json"], the same dict as a JSON string,
for the OpenAI server whose vllm_xargs cannot hold nested dicts; "nolog": true skips the per-step logs):
  {"tag": "<unique id>", "T": 1.0, "ops": [["top_p", 0.9]],
   "ban": [eos ids banned for the whole generation (forced length)],
   "pen": {"presence": 0, "frequency": 0, "repetition": 1.0},
   "max_tokens": N}
Native vLLM sampling must be neutral (temperature=1, top_p=1, top_k=0/-1, min_p=0,
no penalties): this processor applies T, penalties and truncation itself and vLLM
then samples from softmax(processed logits).

Logged per generated position (written to $LCS_STATS_DIR/<tag>.p<k>.npz):
  pos [n] int32, stats [n,6] float32 (samplers.STAT_NAMES)
  tpos [m] int32, lp1 [m] raw log-prob of the sampled token, rank1 [m] its rank under
  the raw distribution (1 = argmax), lq [m] its log-prob under the final sampling
  distribution (finite <=> the token was inside the candidate set; sanity check).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict

import numpy as np
import torch
from vllm.v1.sample.logits_processor import LogitsProcessor, MoveDirectionality

from .samplers import apply_penalties, parse_ops, process_rows

log = logging.getLogger("lcs")
FLUSH_EVERY = 256


def lcs_cfg(params):
    ea = params.extra_args or {}
    cfg = ea.get("lcs")
    if cfg is None and ea.get("lcs_json"):
        cfg = json.loads(ea["lcs_json"])
    return cfg


class RowState:
    __slots__ = ("tag", "T", "ops", "ban", "pen", "out", "max_tokens", "pos", "stats",
                 "tpos", "lp1", "rank1", "lq", "prev", "counts", "ncounted", "done", "sig", "dry", "dry_nout",
                 "prompt")

    def __init__(self, cfg: dict, out: list[int]):
        self.tag = str(cfg["tag"])
        self.T, self.ops = parse_ops(cfg)
        self.ban = [int(x) for x in cfg.get("ban", [])]
        pen = cfg.get("pen") or {}
        self.pen = (float(pen.get("presence", 0.0)), float(pen.get("frequency", 0.0)),
                    float(pen.get("repetition", 1.0)))
        self.out = out
        self.max_tokens = int(cfg.get("max_tokens", 0))
        self.pos, self.stats, self.tpos, self.lp1, self.rank1, self.lq = [], [], [], [], [], []
        self.prev = None  # (step_id, row_in_prev, pos)
        self.counts = None
        self.ncounted = 0
        self.done = bool(cfg.get("nolog"))
        d = cfg.get("dry")
        self.dry = (float(d.get("mult", 0.8)), float(d.get("base", 1.75)), float(d.get("allowed", 2))) if d else None
        self.dry_nout = 0
        self.prompt = None
        self.sig = (tuple(m for m, _ in self.ops), self.pen != (0.0, 0.0, 1.0), self.dry is not None)

    @property
    def has_pen(self):
        return self.pen != (0.0, 0.0, 1.0)


class LCSProcessor(LogitsProcessor):
    @classmethod
    def validate_params(cls, params):
        cfg = lcs_cfg(params)
        if cfg is not None:
            parse_ops(cfg)
            if "tag" not in cfg:
                raise ValueError("lcs config needs a unique 'tag'")

    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = torch.device(device)
        self.rows: dict[int, RowState] = {}
        self.out_dir = os.environ.get("LCS_STATS_DIR", "/tmp/lcs_stats")
        os.makedirs(self.out_dir, exist_ok=True)
        self.step_id = 0
        self.pending: list = []  # (states, positions, stats_gpu)
        self.pending_tok: list = []  # (states, positions, lp1, rank1, lq)
        self.prev_p1 = None
        self.prev_final = None
        self.part = defaultdict(int)
        self.vocab = vllm_config.model_config.get_vocab_size()
        self.vllm_config = vllm_config
        self.dryb = None
        # with tensor parallelism every rank runs this processor; only rank 0 writes logs
        try:
            import torch.distributed as dist
            self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except Exception:
            self.rank = 0
        self._lock = threading.Lock()
        log.warning("LCSProcessor loaded (stats -> %s)", self.out_dir)

    def is_argmax_invariant(self) -> bool:
        return False

    # ------------------------------------------------------------------ batch
    def update_state(self, batch_update):
        if batch_update is None:
            return
        finished = []
        for idx in batch_update.removed:
            st = self.rows.pop(idx, None)
            if st is not None:
                finished.append(st)
        for idx, params, _prompt, out in batch_update.added:
            old = self.rows.pop(idx, None)
            if old is not None:
                finished.append(old)
            cfg = lcs_cfg(params)
            if cfg is not None:
                st = RowState(cfg, out)
                self.rows[idx] = st
                if st.dry is not None:
                    self._dry_buf().init_row(idx, list(_prompt or []) + list(out))
                    st.dry_nout = len(out)
        for src, dst, direction in batch_update.moved:
            if self.dryb is not None:
                self.dryb.move(src, dst, direction == MoveDirectionality.SWAP)
            a, b = self.rows.pop(src, None), self.rows.pop(dst, None)
            if a is not None:
                self.rows[dst] = a
            if b is not None:
                if direction == MoveDirectionality.SWAP:
                    self.rows[src] = b
                else:
                    finished.append(b)  # dst was a finished request overwritten by condensation
        if os.environ.get("LCS_DEBUG"):
            log.warning("update_state removed=%s added=%s moved=%s finished=%s", list(batch_update.removed),
                        [a[0] for a in batch_update.added], list(batch_update.moved), [(s.tag, len(s.pos), len(self.pending)) for s in finished])
        if finished:
            self.flush(finish=finished)

    def _dry_buf(self):
        if self.dryb is None:
            from transformers import AutoTokenizer
            from .dry import DryBuffer, breaker_mask
            mc = self.vllm_config.model_config
            tok = AutoTokenizer.from_pretrained(mc.tokenizer, trust_remote_code=True)
            rows = int(self.vllm_config.scheduler_config.max_num_seqs) + 8
            cap = min(int(mc.max_model_len), 70000)
            self.dryb = DryBuffer(rows, cap, self.vocab, self.device, breaker_mask(tok, self.vocab, self.device))
            log.warning("DRY buffer: %d rows x %d tokens", rows, cap)
        return self.dryb

    # ------------------------------------------------------------------ apply
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.rows:
            self.step_id += 1
            self.prev_p1 = self.prev_final = None
            return logits
        nrows = logits.shape[0]
        items = [(i, st) for i, st in self.rows.items() if i < nrows]
        if not items:
            return logits
        dev = logits.device
        states = [st for _, st in items]
        positions = [len(st.out) for st in states]

        # ---- sampled-token diagnostics for the previous step
        if self.prev_p1 is not None:
            rows_prev, toks, sts, tps = [], [], [], []
            for st, p in zip(states, positions):
                pv = st.prev
                if pv is not None and pv[0] == self.step_id - 1 and p == pv[2] + 1 and p > 0:
                    rows_prev.append(pv[1]); toks.append(st.out[p - 1]); sts.append(st); tps.append(pv[2])
            if rows_prev:
                r = torch.tensor(rows_prev, device=dev, dtype=torch.long)
                t = torch.tensor(toks, device=dev, dtype=torch.long)
                p_rows = self.prev_p1.index_select(0, r)
                pt = p_rows.gather(1, t[:, None])
                rank = (p_rows > pt).sum(-1) + 1
                lp = torch.log(pt.squeeze(1))
                f_rows = self.prev_final.index_select(0, r)
                lq = f_rows.gather(1, t[:, None]).squeeze(1) - torch.logsumexp(f_rows, -1)
                self.pending_tok.append((sts, tps, lp, rank, lq))

        # ---- penalties need per-row output counts
        for st, p in zip(states, positions):
            if st.has_pen:
                if st.counts is None:
                    st.counts = torch.zeros(logits.shape[1], dtype=torch.int32, device=dev)
                if p > st.ncounted:
                    new = torch.tensor(st.out[st.ncounted:p], dtype=torch.long, device=dev)
                    st.counts.index_add_(0, new, torch.ones_like(new, dtype=torch.int32))
                    st.ncounted = p

        if self.dryb is not None:
            todo = {}
            for (i, st), p in zip(items, positions):
                if st.dry is not None and st.dry_nout < p:
                    todo[i] = list(st.out[st.dry_nout:p])
                    st.dry_nout = p
            while todo:
                rr = list(todo)
                self.dryb.push(rr, [todo[i].pop(0) for i in rr])
                todo = {i: v for i, v in todo.items() if v}

        key = tuple(i for i, _ in items) + tuple(id(st) for st in states)
        if key != getattr(self, "_key", None):
            ban_r, ban_c = [], []
            for j, st in enumerate(states):
                for b in st.ban:
                    if b < logits.shape[1]:
                        ban_r.append(j); ban_c.append(b)
            bans = ((torch.tensor(ban_r, device=dev, dtype=torch.long),
                     torch.tensor(ban_c, device=dev, dtype=torch.long)) if ban_r else None)
            groups: dict = defaultdict(list)
            for j, st in enumerate(states):
                groups[st.sig].append(j)
            gl = []
            for sig, js in groups.items():
                sts = [states[j] for j in js]
                jt = torch.tensor(js, device=dev, dtype=torch.long)
                T = torch.tensor([st.T for st in sts], device=dev, dtype=torch.float32)
                all_t1 = all(st.T == 1.0 for st in sts)
                ops = []
                for k, m in enumerate(sig[0]):
                    vals = [st.ops[k][1] for st in sts]
                    t = torch.tensor(vals, device=dev, dtype=torch.float32)
                    if m in ("top_k", "keep_top"):
                        t._kmax = int(max(vals))
                    ops.append((m, t))
                pen = None
                if sig[1]:
                    pen = tuple(torch.tensor([st.pen[q] for st in sts], device=dev) for q in range(3))
                dry = None
                if sig[2]:
                    brow = [items[j][0] for j in js]
                    dry = (brow,) + tuple(torch.tensor([st.dry[q] for st in sts], device=dev) for q in range(3))
                gl.append((jt, sts, T, ops, (pen, dry), all_t1))
            self._key, self._bans, self._groups = key, bans, gl
            bu = sorted({b for st in states for b in st.ban if b < logits.shape[1]})
            self._ban_union = torch.tensor(bu, device=dev, dtype=torch.long) if bu else None
            self._idx_all = torch.tensor([i for i, _ in items], device=dev, dtype=torch.long)

        z_all = logits.index_select(0, self._idx_all).float()
        has_ban = self._bans is not None
        lse_ban = None
        if self._ban_union is not None:
            lse_ban = torch.logsumexp(z_all.index_select(1, self._ban_union), -1)
        if has_ban:
            z_all[self._bans[0], self._bans[1]] = float("-inf")
        if len(self._groups) == 1:
            jt, sts, T, ops, (pen, dry), all_t1 = self._groups[0]
            z = z_all
            if pen is not None:
                z = apply_penalties(z, torch.stack([st.counts for st in sts]), *pen)
            if dry is not None:
                z = z - self.dryb.penalty(*dry)
            out_logits, stats_all, p1_all, _, _ = process_rows(z, T, ops, all_t1, has_ban)
            if lse_ban is not None:
                lse_unb = z.max(-1).values - torch.log(stats_all[:, 2].clamp_min(1e-30))
                stats_all[:, 6] = torch.sigmoid(lse_ban - lse_unb)
        else:
            out_logits = torch.empty_like(z_all)
            stats_all = torch.empty(len(states), 7, device=dev, dtype=torch.float32)
            p1_all = torch.empty_like(z_all)
            for jt, sts, T, ops, (pen, dry), all_t1 in self._groups:
                z = z_all.index_select(0, jt)
                if pen is not None:
                    z = apply_penalties(z, torch.stack([st.counts for st in sts]), *pen)
                if dry is not None:
                    z = z - self.dryb.penalty(*dry)
                final, stats, p1, _, _ = process_rows(z, T, ops, all_t1, has_ban)
                if lse_ban is not None:
                    lse_unb = z.max(-1).values - torch.log(stats[:, 2].clamp_min(1e-30))
                    stats[:, 6] = torch.sigmoid(lse_ban.index_select(0, jt) - lse_unb)
                out_logits.index_copy_(0, jt, final)
                stats_all.index_copy_(0, jt, stats)
                p1_all.index_copy_(0, jt, p1)

        logits.index_copy_(0, self._idx_all, out_logits.to(logits.dtype))
        self.prev_p1 = p1_all
        self.prev_final = out_logits
        for j, (st, p) in enumerate(zip(states, positions)):
            st.prev = (self.step_id, j, p)
        self.pending.append((states, positions, stats_all))
        self.step_id += 1

        last = any(st.max_tokens and p >= st.max_tokens - 1 for st, p in zip(states, positions))
        if last or len(self.pending) >= FLUSH_EVERY:
            self.flush(complete=[st for st, p in zip(states, positions)
                                 if st.max_tokens and p >= st.max_tokens - 1])
        return logits

    # ------------------------------------------------------------------ io
    def flush(self, finish=(), complete=()):
        with self._lock:
            if self.pending:
                big = torch.cat([s for _, _, s in self.pending]).cpu().numpy()
                o = 0
                for sts, poss, s in self.pending:
                    n = len(sts)
                    for k in range(n):
                        st = sts[k]
                        st.pos.append(poss[k]); st.stats.append(big[o + k])
                    o += n
                self.pending = []
            if self.pending_tok:
                lp = torch.cat([x[2] for x in self.pending_tok]).cpu().numpy()
                rk = torch.cat([x[3] for x in self.pending_tok]).cpu().numpy()
                lq = torch.cat([x[4] for x in self.pending_tok]).cpu().numpy()
                o = 0
                for sts, tps, *_ in self.pending_tok:
                    for k in range(len(sts)):
                        st = sts[k]
                        st.tpos.append(tps[k]); st.lp1.append(lp[o + k]); st.rank1.append(rk[o + k]); st.lq.append(lq[o + k])
                    o += len(sts)
                self.pending_tok = []
            for st in list(complete) + list(finish):
                self._write(st)

    def _write(self, st: RowState):
        if getattr(self, "rank", 0) != 0:
            st.pos, st.stats, st.tpos, st.lp1, st.rank1, st.lq = [], [], [], [], [], []
            st.done = True
            return
        if os.environ.get("LCS_DEBUG"):
            log.warning("_write %s done=%s npos=%d", st.tag, st.done, len(st.pos))
        if st.done or not st.pos:
            st.done = True
            return
        k = self.part[st.tag]
        self.part[st.tag] += 1
        path = os.path.join(self.out_dir, f"{st.tag}.p{k}.npz")
        np.savez(path, pos=np.asarray(st.pos, np.int32), stats=np.asarray(st.stats, np.float32),
                 tpos=np.asarray(st.tpos, np.int32), lp1=np.asarray(st.lp1, np.float32),
                 rank1=np.asarray(st.rank1, np.int64), lq=np.asarray(st.lq, np.float32))
        st.pos, st.stats, st.tpos, st.lp1, st.rank1, st.lq = [], [], [], [], [], []
        st.done = True
