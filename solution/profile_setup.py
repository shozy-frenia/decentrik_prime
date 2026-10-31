"""Where the time outside the training steps goes: a sampling profiler of the main thread.

Half of the measured `soup train` time is spent before the first and after the last training
step (imports, Soup's data preparation, model loading, TRL's trainer construction, the final save,
interpreter shutdown). A daemon thread samples the main thread's Python stack every INTERVAL
seconds during those phases (it sleeps without sampling while the steps run) and, when a phase
ends, prints where the samples fell, as `[solution] profile ...` lines in the training log:
  * inclusive: the frames (file:function, or `import <module>`) present in the most samples;
  * self: the innermost frame of each sample (where the interpreter actually was).
Each sample costs ~0.1-0.3 ms of the GIL, i.e. about 1 % of one core at the default interval.
It changes nothing the run computes. Switch: config.SETUP_PROFILE.
"""

import atexit
import collections
import os
import sys
import threading
import time

from . import config

INTERVAL = 0.02
TOP = 28
_LOCK = threading.Lock()
_STATE = {"thread": None, "active": False, "phase": None, "counts": {}, "self": {}, "samples": {},
          "started": {}, "reported": set(), "gc": {}, "gc_t0": None}


def _gc_callback(event, info):
    """Time spent in Python's cyclic GC, per phase and generation (gc.callbacks)."""
    if event == "start":
        _STATE["gc_t0"] = time.perf_counter()
        return
    t0, _STATE["gc_t0"] = _STATE["gc_t0"], None
    if t0 is None:
        return
    rec = _STATE["gc"].setdefault(_STATE["phase"] or "train", {})
    gen = int(info.get("generation", -1))
    n, tot = rec.get(gen, (0, 0.0))
    rec[gen] = (n + 1, tot + time.perf_counter() - t0)


def _short(path):
    path = path.replace("\\", "/")
    for marker in ("/site-packages/", "/dist-packages/", "/lib/python3"):
        at = path.rfind(marker)
        if at >= 0:
            path = path[at + len(marker):]
            break
    return path[-60:]


def _key(frame):
    code = frame.f_code
    if code.co_name == "_find_and_load":
        try:
            return "import " + str(frame.f_locals.get("name"))
        except Exception:  # noqa: BLE001
            return "import ?"
    return f"{_short(code.co_filename)}:{code.co_name}"


def _run(main_ident):
    while True:
        time.sleep(INTERVAL)
        if not _STATE["active"]:
            continue
        try:
            frame = sys._current_frames().get(main_ident)
            phase = _STATE["phase"]
            if frame is None or phase is None:
                continue
            keys, seen, depth, f = [], set(), 0, frame
            while f is not None and depth < 256:
                k = _key(f)
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
                f, depth = f.f_back, depth + 1
            del frame, f
            with _LOCK:
                counts = _STATE["counts"].setdefault(phase, collections.Counter())
                selfc = _STATE["self"].setdefault(phase, collections.Counter())
                _STATE["samples"][phase] = _STATE["samples"].get(phase, 0) + 1
                if keys:
                    selfc[keys[0]] += 1
                counts.update(keys)
        except Exception:  # noqa: BLE001 - a diagnostic never disturbs the run
            continue


def start():
    """Called first thing in apply(): the 'setup' phase starts now."""
    if not config.SETUP_PROFILE or _STATE["thread"] is not None:
        return
    main = threading.main_thread().ident
    t = threading.Thread(target=_run, args=(main,), daemon=True, name="soup-setup-profile")
    _STATE["thread"] = t
    phase("setup")
    t.start()
    try:
        import gc
        gc.callbacks.append(_gc_callback)
    except Exception:  # noqa: BLE001
        pass
    atexit.register(_at_exit)


def phase(name):
    """Start sampling into `name` (None pauses)."""
    if _STATE["thread"] is None and name is not None and not config.SETUP_PROFILE:
        return
    _STATE["phase"] = name
    _STATE["active"] = name is not None
    if name is not None:
        _STATE["started"].setdefault(name, time.perf_counter())


def report(name):
    """Print the profile of phase `name` once (never raises)."""
    try:
        _report(name)
    except Exception as exc:  # noqa: BLE001
        print(f"[solution] profile {name}: not available ({type(exc).__name__}: {exc})",
              file=sys.stderr, flush=True)


def _report(name):
    if name in _STATE["reported"] or name not in _STATE["samples"]:
        return
    _STATE["reported"].add(name)
    with _LOCK:
        n = _STATE["samples"][name]
        tables = {"inclusive": collections.Counter(_STATE["counts"][name]),
                  "self": collections.Counter(_STATE["self"][name])}
    wall = time.perf_counter() - _STATE["started"].get(name, time.perf_counter())
    out = sys.stderr
    print(f"[solution] profile {name}: {n} samples x {INTERVAL * 1000:.0f} ms over {wall:.1f} s "
          f"(pid {os.getpid()}); seconds = samples x interval", file=out, flush=True)
    for title, counter in tables.items():
        items = [(k, c) for k, c in counter.most_common(TOP * 2)
                 if not (title == "inclusive" and c >= 0.97 * n)][:TOP]
        line = "; ".join(f"{k} {c * INTERVAL:.2f}" for k, c in items)
        print(f"[solution] profile {name} {title}: {line}", file=out, flush=True)
    gcs = _STATE["gc"].get(name, {})
    if gcs:
        text = ", ".join(f"gen{g}: {n} in {t:.2f} s" for g, (n, t) in sorted(gcs.items()))
        print(f"[solution] profile {name} gc: {text}", file=out, flush=True)


def _at_exit():
    phase(None)
    report("setup")
    report("wrapup")
    gcs = _STATE["gc"].get("train", {})
    if gcs:
        text = ", ".join(f"gen{g}: {n} in {t:.2f} s" for g, (n, t) in sorted(gcs.items()))
        print(f"[solution] profile train gc: {text}", file=sys.stderr, flush=True)
