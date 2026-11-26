"""Numeric image-side quality flags for MEBeauty, calibrated on SCUT (NO ratings of either set are read; written before the run).
Flags (each fixed here, before any MEBeauty label is read):
  sd      at least 5 of the 70 tier-2 columns with |z| > 3 inside the group (from mebeauty_flags_sd.csv, registered 2026-11-07)
  roll    |roll_deg| above the 99th percentile of SCUT non-lockbox faces
  yaw     |half_ratio - 1| above the 99th percentile of SCUT (half-face ratio: head turned away from the camera)
  mouth   mouth_open above the 99th percentile of SCUT (open mouth)
  hair    trichion not found (hairline walk failed: hat, hood, hair over the forehead, or a cut-off forehead)
  any     at least one of the above
NOT covered (no detector, stated openly): glasses, beards (partly in sd through chin_sd), children, heavy make-up.
Reported: the SCUT thresholds, the share of SCUT faces flagged by each rule (the 99th-percentile rules give 1 % by construction; 'hair' about 3 %), the share of MEBeauty faces flagged by each rule and by any, per group.
Expectation (AI, before): MEBeauty flagged shares: roll 3-8 %, yaw 3-8 %, mouth 5-15 % (stock photos smile), hair 5-6 %, any 15-30 %.
Output: data/external/mebeauty/mebeauty_flags.csv (file, flag_sd, flag_roll, flag_yaw, flag_mouth, flag_hair, flag_any). Usage: python mebeauty_flags.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mebeauty_feature_audit as AU
import overnight_common as O

R = O.R
MB = R / "data/external/mebeauty"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df, props, rest, tier2 = O.load()
    sc = AU.scut_raw(tier2)
    mb = pd.read_csv(MB / "mebeauty_features.csv")
    mb["group"] = mb.ethnicity + "_" + mb.gender
    sd = pd.read_csv(MB / "mebeauty_flags_sd.csv")[["file", "flag"]].rename(columns={"flag": "flag_sd"})
    mb = mb.merge(sd, on="file", how="left")
    sc["half_dev2"] = (sc.half_ratio - 1).abs()
    mb["half_dev2"] = (mb.half_ratio - 1).abs()
    sc["roll_abs2"] = sc.roll_deg.abs()
    mb["roll_abs2"] = mb.roll_deg.abs()
    th = {"roll": sc.roll_abs2.quantile(0.99), "yaw": sc.half_dev2.quantile(0.99), "mouth": sc.mouth_open.quantile(0.99)}
    print("SCUT 99th percentile thresholds:", {k: round(float(v), 4) for k, v in th.items()})
    sc_flags = {"roll": sc.roll_abs2 > th["roll"], "yaw": sc.half_dev2 > th["yaw"], "mouth": sc.mouth_open > th["mouth"], "hair": sc.trichion_found == 0}
    mb["flag_roll"] = mb.roll_abs2 > th["roll"]
    mb["flag_yaw"] = mb.half_dev2 > th["yaw"]
    mb["flag_mouth"] = mb.mouth_open > th["mouth"]
    mb["flag_hair"] = mb.trichion_found == 0
    mb["flag_any"] = mb[["flag_sd", "flag_roll", "flag_yaw", "flag_mouth", "flag_hair"]].fillna(False).astype(bool).any(axis=1)
    print("\nshare flagged            SCUT      MEBeauty")
    for k in ("roll", "yaw", "mouth", "hair"):
        print(f"  {k:6s}              {100 * sc_flags[k].mean():5.1f} %   {100 * mb['flag_' + k].mean():5.1f} %")
    print(f"  sd                    (2.5 %)   {100 * mb.flag_sd.mean():5.1f} %")
    print(f"  any                             {100 * mb.flag_any.mean():5.1f} %  ({int(mb.flag_any.sum())} of {len(mb)} faces)")
    print("\nflagged by any rule, per group:")
    print(mb.groupby("group").flag_any.agg(["size", "mean"]).round(3).to_string())
    mb[["file", "flag_sd", "flag_roll", "flag_yaw", "flag_mouth", "flag_hair", "flag_any"]].to_csv(MB / "mebeauty_flags.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
