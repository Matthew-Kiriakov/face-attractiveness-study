// How each of the 42 proportions (js/metrics.js) responds to a small move of each drawn landmark, at every set's mean shape.
// Called by build_data.py: node jacobian.mjs in.json out.json
import { readFileSync, writeFileSync } from "node:fs";
import { measure } from "../js/metrics.js";

const [inp, out] = process.argv.slice(2);
const { index, model, props, sets } = JSON.parse(readFileSync(inp, "utf8"));
const pos = { trichion: index.indexOf("trichion"), glabella: index.indexOf("glabella") };
for (const k in model) pos[k] = index.indexOf(model[k]);

function read(P) {
  const q = {};
  for (const k in pos) q[k] = { x: P[pos[k]][0], y: P[pos[k]][1] };
  const m = measure(q);
  return props.map((p) => m[p]);
}

const h = 1e-4, J = {}, at = {};
for (const s in sets) {
  const P = sets[s];
  at[s] = read(P);
  J[s] = props.map(() => []);
  P.forEach((_, i) => [0, 1].forEach((a) => {
    const up = P.map((p) => [...p]), dn = P.map((p) => [...p]);
    up[i][a] += h;
    dn[i][a] -= h;
    const u = read(up), d = read(dn);
    props.forEach((_, j) => J[s][j].push((u[j] - d[j]) / (2 * h)));
  }));
}
writeFileSync(out, JSON.stringify({ J, at }));
