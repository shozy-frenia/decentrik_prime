"""importlib.metadata.packages_distributions(), computed once per process.

transformers computes the {top-level package: [distributions]} map when it is imported (before
apply(), in the harness); TRL's import_utils computes the same map again when Soup imports TRL
during its setup, which reads the metadata of every installed distribution (2.4 s of file reads
on Kaggle's disk, in the v8 setup profile). From apply() until the first training step, the
function returns a copy of transformers' map instead: the set of installed packages cannot change
in between, so every caller gets exactly what it would have computed. The original function is
restored at the first step (and at exit, whatever happens).
"""

import atexit

_STATE = {"orig": None}


def share():
    import importlib.metadata as md

    if _STATE["orig"] is not None:
        return
    try:
        from transformers.utils import import_utils as iu

        cached = iu.PACKAGE_DISTRIBUTION_MAPPING
        if not isinstance(cached, dict) or not cached:
            return
    except Exception:  # noqa: BLE001 - compute it as usual
        return
    orig = md.packages_distributions

    def packages_distributions():
        return {k: list(v) for k, v in cached.items()}

    packages_distributions.__wrapped__ = orig
    _STATE["orig"] = orig
    md.packages_distributions = packages_distributions
    atexit.register(restore)


def restore():
    orig, _STATE["orig"] = _STATE["orig"], None
    if orig is not None:
        import importlib.metadata as md

        md.packages_distributions = orig
