"""One-off: crops the MakeHuman base mesh (CC0, see LICENSES.md) to head + neck + eyeballs and writes page/assets/head.js,
a compact asset (int16 positions, uint16 quads) plus the control vertices the page warps with its landmark RBF.
Usage: python page/build_head.py   (reads page/assets/base.obj and page/data.json; the full body mesh is not shipped)."""
import base64
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

PAGE = Path(__file__).resolve().parent
YCUT = 5.85
SETS = ["CM 10k_US", "CF 10k_US", "AM DataTang", "AM internet", "AF DataTang", "AF internet"]

V, G, cur = [], OrderedDict(), None
for ln in open(PAGE / "assets/base.obj"):
    if ln.startswith("v "):
        V.append([float(x) for x in ln.split()[1:4]])
    elif ln.startswith("g "):
        cur = ln.split()[1]
        G.setdefault(cur, [])
    elif ln.startswith("f "):
        G[cur].append([int(t.split("/")[0]) - 1 for t in ln.split()[1:]])
V = np.array(V)

body = np.array(G["body"])
head_q = body[(V[body][:, :, 1] > YCUT).all(axis=1)]
eyes_q = {s: np.array(G[f"helper-{s}-eye"]) for s in "lr"}
used = np.unique(np.concatenate([head_q.ravel()] + [q.ravel() for q in eyes_q.values()]))
remap = -np.ones(len(V), int)
remap[used] = np.arange(len(used))
P = V[used]
head_q, eyes_q = remap[head_q], {s: remap[q] for s, q in eyes_q.items()}
n_head = len(np.unique(head_q))
is_head = np.zeros(len(P), bool)
is_head[np.unique(head_q)] = True

# landmark layout of the page: mean of the six set means, in the page's units (x right, y down, hairline-to-chin = 100)
D = json.load(open(PAGE / "data.json", encoding="utf-8"))
ix = D["face"]["points"]["index"]
M = np.mean([np.array(D["face"]["sets"][s]["mean"]) * 100 for s in SETS], axis=0)
lm = lambda i: M[ix.index(i)]

# similarity landmark -> mesh frontal plane: eye centres, nose tip, mouth centre
eye_c = {s: P[np.unique(eyes_q[s])].mean(axis=0) for s in "lr"}
tip = P[is_head & (np.abs(P[:, 0]) < .1) & (P[:, 1] > 6.7) & (P[:, 1] < 7.1)]
tip = tip[tip[:, 2].argmax()]
teeth = V[np.unique(np.array(G["helper-upper-teeth"] + G["helper-lower-teeth"]).ravel())]
mouth_y = (teeth[:, 1].min() + teeth[:, 1].max()) / 2
pairs = [(lm(468), (eye_c["r"][0], eye_c["r"][1])), (lm(473), (eye_c["l"][0], eye_c["l"][1])), (lm(1), (tip[0], tip[1])),
         ((M[ix.index(13)] + M[ix.index(14)]) / 2, (0, mouth_y))]
A = np.array([[p[0], 1, 0] for p, _ in pairs] + [[-p[1], 0, 1] for p, _ in pairs])
b = np.array([m[0] for _, m in pairs] + [m[1] for _, m in pairs])
(s, tx, ty), *_ = np.linalg.lstsq(A, b, rcond=None)
res = A @ [s, tx, ty] - b
print(f"similarity: scale {s:.5f}, t ({tx:.3f}, {ty:.3f}); residuals max {np.abs(res).max():.3f} (face width ~1.4)")

# control vertices: nearest front-facing head vertex in the frontal plane for every landmark, one landmark per vertex
nrm = np.zeros_like(P)
for q in head_q:
    n = np.cross(P[q[1]] - P[q[0]], P[q[3]] - P[q[0]])
    nrm[q] += n
nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
front = np.where(is_head & (nrm[:, 2] > .2))[0]
taken, ctrl = set(), []
for i in ix:
    p = lm(i)
    q = np.array([s * p[0] + tx, -s * p[1] + ty])
    d = np.linalg.norm(P[front][:, :2] - q, axis=1)
    for j in np.argsort(d):
        if front[j] not in taken:
            taken.add(front[j])
            ctrl.append(int(front[j]))
            break
ctrl = np.array(ctrl)
print("control vertices", len(ctrl), "mean offset from landmark", np.linalg.norm(P[ctrl][:, :2] - np.array([[s * lm(i)[0] + tx, -s * lm(i)[1] + ty] for i in ix]), axis=1).mean().round(3))

# anchors that never move: neck base and cranium top
heads = np.where(is_head)[0]
anch = np.concatenate([heads[P[heads, 1] < YCUT + .12][::3], heads[P[heads, 1] > 8.3][::4]])
anch = np.setdiff1d(anch, ctrl)
print("anchors", len(anch))

lo, hi = P.min(0), P.max(0)
sc = (hi - lo).max() / 65534
q16 = np.round((P - lo) / sc - 32767).astype(np.int16)
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
out = {"n": len(P), "pos": b64(q16), "lo": lo.tolist(), "sc": float(sc), "head": b64(head_q.astype(np.uint16)), "eyeL": b64(eyes_q["l"].astype(np.uint16)),
       "eyeR": b64(eyes_q["r"].astype(np.uint16)), "ctrl": ctrl.tolist(), "anch": anch.tolist(), "T": [float(s), float(tx), float(ty)],
       "eyes": {"l": eye_c["l"].tolist(), "r": eye_c["r"].tolist()}, "baseY": YCUT}
(PAGE / "assets/head.js").write_text("window.HEAD=" + json.dumps(out, separators=(",", ":")) + ";\n")
print("vertices", len(P), "head quads", len(head_q), "file KB", round((PAGE / "assets/head.js").stat().st_size / 1024))
