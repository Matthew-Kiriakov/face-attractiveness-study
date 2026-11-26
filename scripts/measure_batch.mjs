// Same formulas as the page: metrics.js measure() and gates.js.
// Input JSONL: {file, w, h, lm:[[x,y],...], trichion:[x,y]|null, glabella:[x,y]|null, trichion_found, error?}
// Coordinates in lm / trichion / glabella are normalized, as in the browser.

import fs from "fs";
import { measure, METRICS, poseOf, POINTS } from "../js/metrics.js";
import { GATES, gateHolds } from "../js/gates.js";

const inputPath = process.argv[2];
const outputPath = process.argv[3];
const lines = fs.readFileSync(inputPath, "utf8").split(/\n/).filter((line) => line.trim());
const rows = [];

for (const line of lines) {
  const row = JSON.parse(line);
  if (row.error) {
    rows.push({ file: row.file, error: row.error });
    continue;
  }
  const points = {};
  let missing = null;
  for (const spec of POINTS) {
    const lm = row.lm[spec.index];
    if (!lm) {
      missing = spec.id;
      break;
    }
    points[spec.id] = { x: lm[0] * row.w, y: lm[1] * row.h, index: spec.index };
  }
  if (missing) {
    rows.push({ file: row.file, error: `missing ${missing}` });
    continue;
  }
  if (row.trichion) {
    points.trichion = { x: row.trichion[0] * row.w, y: row.trichion[1] * row.h };
  }
  if (row.glabella) {
    points.glabella = { x: row.glabella[0] * row.w, y: row.glabella[1] * row.h };
  }
  const values = measure(points);
  const pose = poseOf(points);
  let ticks = 0;
  let ticksOf = 0;
  if (values) {
    for (const gate of GATES) {
      const held = gateHolds(gate, values[gate.id]);
      if (held == null) continue;
      ticksOf += 1;
      if (held) ticks += 1;
    }
  }
  rows.push({
    file: row.file,
    trichion_found: row.trichion_found ? 1 : 0,
    roll: pose.roll,
    half: pose.half,
    ticks,
    ticks_of: ticksOf,
    values,
  });
}

const catalog = METRICS.map((metric) => ({
  id: metric.id,
  group: metric.group,
  name: metric.name,
}));
fs.writeFileSync(outputPath, JSON.stringify({ catalog, rows }));
