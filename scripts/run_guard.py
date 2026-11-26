"""Run guard for the two one-shot confirmatory runs (FREEZE v1.4): input hashes, marker, .part output.
Used by scut_lockbox_S.py and mebeauty_B.py in mode 'run' only.  Usage to (re)write the registered hashes (before the freeze commit, never after):
  python run_guard.py freeze        -> writes scripts/input_hashes.json
  python run_guard.py check         -> prints OK or the files that differ
"""
from __future__ import annotations

import glob
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

R = Path(__file__).resolve().parent.parent
HASHFILE = Path(__file__).resolve().parent / "input_hashes.json"
FIXED = [
    "data/scut_cm_harmony.csv", "data/scut_am_harmony.csv", "data/scut_cf_harmony.csv", "data/scut_af_harmony.csv",
    "data/scut_cm_quality.csv", "data/scut_cm_source.csv", "data/scut_age_mesh_all.csv", "data/lockbox.csv",
    "data/scut_colour_features.csv", "data/scut_block2b_features.csv", "data/scut_block2c_features.csv",
    "data/scut_block3_symmetry.csv", "data/scut_block5_dimorphism.csv", "data/scut_block6_angularity.csv",
    "data/scut_embedding.npz", "data/scut-fbp5500/SCUT-FBP5500_v2/All_Ratings.xlsx",
    "data/external/mebeauty/mebeauty_features.csv", "data/external/mebeauty/mebeauty_flags.csv",
    "data/external/mebeauty/mebeauty_landmarks_web.jsonl",
]
GLOBS = ["data/external/mebeauty/fbp_extended/*.parquet"]


def files() -> list[str]:
    out = list(FIXED)
    for g in GLOBS:
        out += sorted(str(Path(p).relative_to(R)).replace("\\", "/") for p in glob.glob(str(R / g)))
    return out


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def current() -> dict[str, str]:
    return {f: (sha(R / f) if (R / f).exists() else "MISSING") for f in files()}


def versions() -> dict[str, str]:
    import numpy, pandas, scipy, sklearn
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def verify() -> None:
    reg = json.loads(HASHFILE.read_text(encoding="utf-8"))
    now = current()
    bad = [f for f in sorted(set(reg["inputs"]) | set(now)) if reg["inputs"].get(f) != now.get(f)]
    if bad:
        sys.exit("REFUSED before the marker: input files differ from scripts/input_hashes.json: " + ", ".join(bad))
    if reg["versions"] != versions():
        sys.exit(f"REFUSED before the marker: library versions differ from the registered ones: registered {reg['versions']}, now {versions()}")


def start(done: Path, results: Path, script: Path, rerun: bool, step: str) -> Path:
    """Verifies inputs, then creates the marker (or records ONE rerun after a crash). Returns the path of the .part file."""
    verify()
    part = results.with_suffix(".txt.part")
    meta = {"step": step, "started": datetime.now().isoformat(timespec="seconds"), "script_sha256": sha(script), "versions": versions()}
    if rerun:
        if not done.exists() or results.exists():
            sys.exit("REFUSED: --rerun is only for a run that started (marker exists) and did not finish (no results file).")
        if "RERUN" in done.read_text(encoding="utf-8"):
            sys.exit("REFUSED: the one allowed rerun was already used; a second failure ends the confirmatory status of this step.")
        with done.open("a", encoding="utf-8") as f:
            f.write("RERUN " + json.dumps(meta) + "\n")
        if part.exists():
            part.replace(part.with_suffix(".part.crashed"))  # not to be opened by the author or the agent
    else:
        if done.exists():
            sys.exit(f"REFUSED: {done.name} exists; the run is done once. After a crash see FREEZE v1.4 (--rerun).")
        done.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    part.write_text("", encoding="utf-8")
    return part


class Out:
    """Collects lines; in run mode numbers go to the .part file (flushed per line), the console gets stage names only."""

    def __init__(self, part: Path | None):
        self.lines: list[str] = []
        self.part = part
        self.f = part.open("a", encoding="utf-8") if part else None

    def say(self, s: str = "") -> None:
        self.lines.append(s)
        if self.f:
            self.f.write(s + "\n")
            self.f.flush()
        else:
            print(s)

    def stage(self, name: str) -> None:
        print(f"[stage] {name}", flush=True)

    def finish(self, final: Path) -> None:
        if self.f:
            self.f.close()
            self.part.replace(final)


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        if HASHFILE.exists():
            sys.exit("input_hashes.json exists; it is written once (delete only before the freeze commit).")
        HASHFILE.write_text(json.dumps({"inputs": current(), "versions": versions()}, indent=1), encoding="utf-8")
        print("written", HASHFILE, len(files()), "files")
    elif sys.argv[1] == "check":
        verify()
        print("OK: inputs and versions match the registered ones")
