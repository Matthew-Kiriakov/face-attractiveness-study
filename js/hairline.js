// Walk up the forehead until the colour leaves the skin and stays there.
// Coordinates are whatever the caller uses; `at(i)` returns the i-th sample
// upward, or null past the image.

const COLOR_GAP = 32;
const LUMA_DROP = 14;
const RUN = 5;

function luma(pixel) {
  return 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;
}

function gap(pixel, skin) {
  return Math.hypot(pixel.r - skin.r, pixel.g - skin.g, pixel.b - skin.b);
}

export function findHairline({ at, skin, maxSteps }) {
  let run = 0;
  let start = null;
  const skinLuma = luma(skin);
  for (let i = 1; i <= maxSteps; i += 1) {
    const sample = at(i);
    if (!sample) break;
    const unlike = gap(sample, skin) > COLOR_GAP && skinLuma - luma(sample) > LUMA_DROP;
    if (unlike) {
      if (run === 0) start = sample;
      run += 1;
      if (run >= RUN) return { x: start.x, y: start.y };
    } else {
      run = 0;
      start = null;
    }
  }
  return null;
}
