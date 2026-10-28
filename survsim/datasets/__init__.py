"""Dataset packaging: seeded runs (or MOTChallenge sequences) -> one manifest.

The manifest is the only thing `survsim baseline` reads, so SurvSim renders and
real MOT20 footage are scored by exactly the same code path.
"""
from .build import build_dataset, from_motchallenge, load_manifest, read_gt, split_for_seed

__all__ = ["build_dataset", "from_motchallenge", "load_manifest", "read_gt",
           "split_for_seed"]
