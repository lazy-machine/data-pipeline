"""data-verify: a leakage and poisoning gate for data-pipeline projects.

In a pipeline file::

    [[steps]]
    name = "verify"
    function = "data_verify:verify"
    inputs = { train = "build/train.parquet", test = "build/test.parquet" }
    outputs = { report = "build/verify/report.json" }
    params = { label = "y", group = "customer_id" }

The step fails (and blocks every step that depends on it) when a blocking check fails.
Bump __version__ on every change: data-pipeline fingerprints this file, so the gate reruns.
"""

__version__ = "0.1.0"

from .core import Verifier  # noqa: E402
from .spec import SpecError  # noqa: E402
from .step import VerificationFailed, run, verify  # noqa: E402

__all__ = ["SpecError", "VerificationFailed", "Verifier", "__version__", "run", "verify"]
