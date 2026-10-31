"""GPU-batched incremental DRY ("Don't Repeat Yourself") penalty (Weidmann et al., 2026).

For candidate token t, L(t) = length of the longest suffix of the current context that also
occurred earlier immediately followed by t.  DRY subtracts
    multiplier * base ** (L(t) - allowed_length)     if L(t) >= allowed_length.
Matches do not extend across sequence-breaker tokens.

State per row (persistent buffer indexed by vLLM batch row): ctx[0..n-1] and
M[j] = length of the common suffix of ctx[:j] and ctx[:n].  Appending x:
    M'[j] = M[j-1] + 1  if ctx[j-1] == x and x is not a breaker,  else 0      (j = 1..n)
an O(n) vector op per step, batched over rows.
"""
from __future__ import annotations

import torch

DEFAULT_BREAKERS = ("\n", ":", '"', "*")


def breaker_mask(tokenizer, vocab_size, device, breakers=DEFAULT_BREAKERS):
    m = torch.zeros(vocab_size, dtype=torch.bool)
    n = min(vocab_size, len(tokenizer))
    pieces = tokenizer.batch_decode([[i] for i in range(n)])
    for i, s in enumerate(pieces):
        if any(b in s for b in breakers):
            m[i] = True
    return m.to(device)


class DryBuffer:
    def __init__(self, rows, cap, vocab, device, is_breaker):
        self.cap, self.V, self.device = cap, vocab, device
        self.ctx = torch.zeros(rows, cap, dtype=torch.long, device=device)
        self.M = torch.zeros(rows, cap, dtype=torch.int32, device=device)
        self.n = [0] * rows
        self.is_breaker = is_breaker
        self.ar = torch.arange(cap, device=device)

    def reset(self, r):
        self.n[r] = 0

    def move(self, src, dst, swap):
        if swap:
            self.ctx[[src, dst]] = self.ctx[[dst, src]]
            self.M[[src, dst]] = self.M[[dst, src]]
            self.n[src], self.n[dst] = self.n[dst], self.n[src]
        else:
            self.ctx[dst] = self.ctx[src]
            self.M[dst] = self.M[src]
            self.n[dst] = self.n[src]

    def init_row(self, r, tokens):
        """Set row r to `tokens` (prompt + generated so far), computing M in one shot."""
        n = min(len(tokens), self.cap)
        self.n[r] = n
        if n == 0:
            return
        c = torch.tensor(tokens[:n], device=self.device, dtype=torch.long)
        self.ctx[r, :n] = c
        self.M[r].zero_()
        if n < 2:
            return
        if n <= 8192:
            # E[j, i] = c[j-1-i] == c[n-1-i] and c[n-1-i] not a breaker, for j=1..n-1, i < j
            j = torch.arange(1, n, device=self.device)[:, None]
            i = torch.arange(n, device=self.device)[None, :]
            ok = i < j
            a = c[(j - 1 - i).clamp_min(0)]
            b = c[(n - 1 - i).clamp_min(0)]
            E = ok & (a == b) & ~self.is_breaker[b]
            self.M[r, 1:n] = E.int().cumprod(-1).sum(-1).to(torch.int32)
        else:
            # M[j] = common-suffix length of c[:j] and c = LCP(rev(c), rev(c)[n-j:]) = Z[n-j] on the reversed
            # sequence, capped at the first breaker of the current suffix (and at j).  O(n) Z-algorithm.
            rc = [int(x) for x in reversed(tokens[:n])]
            brk = self.is_breaker.cpu().numpy() if not hasattr(self, "_brk_np") else self._brk_np
            self._brk_np = brk
            cap = n
            for i, x in enumerate(rc):
                if x < len(brk) and brk[x]:
                    cap = i
                    break
            Z = [0] * n
            lft = rgt = 0
            for i in range(1, n):
                if i < rgt:
                    Z[i] = min(rgt - i, Z[i - lft])
                while i + Z[i] < n and rc[Z[i]] == rc[i + Z[i]]:
                    Z[i] += 1
                if i + Z[i] > rgt:
                    lft, rgt = i, i + Z[i]
            js = list(range(1, n))
            vals = [min(Z[n - j], cap, j) for j in js]
            self.M[r, 1:n] = torch.tensor(vals, dtype=torch.int32, device=self.device)

    def push(self, rows, toks):
        """Append one token per row (rows: list[int], toks: list[int])."""
        if not rows:
            return
        r = torch.tensor(rows, device=self.device)
        x = torch.tensor(toks, device=self.device)
        N = torch.tensor([self.n[i] for i in rows], device=self.device)
        ctx = self.ctx.index_select(0, r)
        M = self.M.index_select(0, r)
        valid = self.ar[None, :] < N[:, None]
        eq = (ctx == x[:, None]) & valid & ~self.is_breaker[x][:, None]
        newM = torch.zeros_like(M)
        newM[:, 1:] = torch.where(eq[:, :-1], M[:, :-1] + 1, torch.zeros_like(M[:, :-1]))
        ctx.scatter_(1, N.clamp_max(self.cap - 1)[:, None], x[:, None])
        self.ctx.index_copy_(0, r, ctx)
        self.M.index_copy_(0, r, newM)
        for i in rows:
            self.n[i] = min(self.n[i] + 1, self.cap)

    def penalty(self, rows, mult, base, allowed):
        """[len(rows), V] penalties.  mult/base/allowed: float tensors [len(rows)]."""
        r = torch.tensor(rows, device=self.device)
        N = torch.tensor([self.n[i] for i in rows], device=self.device)
        M = self.M.index_select(0, r)[:, 1:].to(torch.float32)
        nxt = self.ctx.index_select(0, r)[:, 1:]
        valid = self.ar[None, 1:] < N[:, None]          # j in [1, n-1]
        L = torch.where(valid, M, torch.zeros_like(M))
        best = torch.zeros(len(rows), self.V, device=self.device)
        best.scatter_reduce_(1, torch.where(valid, nxt, torch.zeros_like(nxt)), L, reduce="amax")
        al = allowed[:, None]
        pen = torch.where(best >= al, mult[:, None] * torch.pow(base[:, None], best - al), torch.zeros_like(best))
        return pen.masked_fill(self.is_breaker[None, :pen.shape[1]], 0.0)
