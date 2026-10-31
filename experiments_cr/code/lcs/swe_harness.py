"""Run the official SWE-bench harness with one robustness patch: listing containers (used only for the end-of-run
report) retries when a container disappears mid-listing, which happens constantly while hundreds of agent containers
are being created and removed on the same host.  Grading itself is unchanged.
python -m lcs.swe_harness <run_evaluation args>
"""
import runpy
import time

import docker
from docker.models.containers import ContainerCollection

_orig = ContainerCollection.list


def _robust_list(self, *a, **k):
    for _ in range(30):
        try:
            return _orig(self, *a, **k)
        except docker.errors.NotFound:
            time.sleep(1)
    return []


ContainerCollection.list = _robust_list

if __name__ == "__main__":
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__", alter_sys=True)
