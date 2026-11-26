// Frontal measurements from MediaPipe Face Landmarker indices.
// Left and right are the sides of the photo, not anatomical sides.
// Indices follow the face-mesh graph: image-left eye is 33/133, image-right is 263/362.

export const POINTS = [
  { id: "forehead", index: 10, label: "лоб" },
  { id: "menton", index: 152, label: "подбородок" },
  { id: "nasion", index: 168, label: "назион" },
  { id: "pronasale", index: 1, label: "кончик" },
  { id: "subnasale", index: 2, label: "субназале" },
  { id: "bridgeUpper", index: 6, label: "спинка" },
  { id: "bridgeMid", index: 197, label: "спинка" },
  { id: "bridgeLower", index: 195, label: "спинка" },
  { id: "alarL", index: 98, label: "крыло" },
  { id: "alarR", index: 327, label: "крыло" },
  { id: "nostrilTopL", index: 45, label: "ноздря" },
  { id: "nostrilLowL", index: 48, label: "ноздря" },
  { id: "nostrilTopR", index: 275, label: "ноздря" },
  { id: "nostrilLowR", index: 278, label: "ноздря" },
  { id: "eyeOuterL", index: 33, label: "наружный угол" },
  { id: "eyeInnerL", index: 133, label: "внутренний угол" },
  { id: "eyeTopL", index: 159, label: "верх века" },
  { id: "eyeBotL", index: 145, label: "низ века" },
  { id: "irisL", index: 468, label: "зрачок" },
  { id: "eyeOuterR", index: 263, label: "наружный угол" },
  { id: "eyeInnerR", index: 362, label: "внутренний угол" },
  { id: "eyeTopR", index: 386, label: "верх века" },
  { id: "eyeBotR", index: 374, label: "низ века" },
  { id: "irisR", index: 473, label: "зрачок" },
  { id: "browOuterL", index: 46, label: "хвост брови" },
  { id: "browInnerL", index: 107, label: "головка брови" },
  { id: "browOuterR", index: 276, label: "хвост брови" },
  { id: "browInnerR", index: 336, label: "головка брови" },
  { id: "mouthL", index: 61, label: "угол рта" },
  { id: "mouthR", index: 291, label: "угол рта" },
  { id: "cupid", index: 0, label: "лук Купидона" },
  { id: "cupidL", index: 37, label: "пик лука" },
  { id: "cupidR", index: 267, label: "пик лука" },
  { id: "lipUpper", index: 13, label: "смыкание" },
  { id: "lipLower", index: 14, label: "смыкание" },
  { id: "lipBot", index: 17, label: "низ губы" },
  { id: "zygionL", index: 234, label: "край лица" },
  { id: "zygionR", index: 454, label: "край лица" },
  { id: "gonionL", index: 172, label: "угол челюсти" },
  { id: "gonionR", index: 397, label: "угол челюсти" },
  { id: "templeL", index: 54, label: "висок" },
  { id: "templeR", index: 284, label: "висок" },
];

const LABEL = Object.fromEntries(POINTS.map((p) => [p.id, p.label]));
LABEL.browL = "середина брови";
LABEL.browR = "середина брови";
LABEL.stomion = "щель рта";
LABEL.pupils = "зрачки";
LABEL.trichion = "линия роста";
LABEL.glabella = "между бровями";
export { LABEL };

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function mid(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function faceFrame(p) {
  const o = p.nasion;
  const vx = p.menton.x - o.x;
  const vy = p.menton.y - o.y;
  const len = Math.hypot(vx, vy) || 1;
  const v = { x: vx / len, y: vy / len };
  const u = { x: v.y, y: -v.x };
  return { o, u, v, axis: len };
}

function xy(frame, pt) {
  const dx = pt.x - frame.o.x;
  const dy = pt.y - frame.o.y;
  return {
    u: dx * frame.u.x + dy * frame.u.y,
    v: dx * frame.v.x + dy * frame.v.y,
  };
}

function ratio(a, b) {
  if (!b) return null;
  const v = a / b;
  return Number.isFinite(v) ? v : null;
}

function percent(part, whole) {
  const v = ratio(part, whole);
  return v == null ? null : v * 100;
}

function topOfFace(p) {
  return p.trichion || p.forehead;
}

function faceSpan(p) {
  const top = topOfFace(p);
  if (!top || !p.menton) return null;
  return dist(top, p.menton);
}

function tilt(frame, medial, lateral) {
  const a = xy(frame, medial);
  const b = xy(frame, lateral);
  const du = Math.abs(b.u - a.u);
  const dv = a.v - b.v;
  return (Math.atan2(dv, du) * 180) / Math.PI;
}

function incline(frame, a, b) {
  const A = xy(frame, a);
  const B = xy(frame, b);
  return (Math.atan2(Math.abs(B.v - A.v), Math.abs(B.u - A.u)) * 180) / Math.PI;
}

function angleAt(a, vertex, b) {
  const v1x = a.x - vertex.x;
  const v1y = a.y - vertex.y;
  const v2x = b.x - vertex.x;
  const v2y = b.y - vertex.y;
  const d1 = Math.hypot(v1x, v1y);
  const d2 = Math.hypot(v2x, v2y);
  if (!d1 || !d2) return null;
  const cos = Math.min(1, Math.max(-1, (v1x * v2x + v1y * v2y) / (d1 * d2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function along(a, b, pt) {
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const len2 = vx * vx + vy * vy;
  if (!len2) return null;
  return ((pt.x - a.x) * vx + (pt.y - a.y) * vy) / len2;
}

function thirdBetween(points, from, to) {
  if (!points.trichion) return null;
  const start = along(points.trichion, points.menton, from);
  const end = along(points.trichion, points.menton, to);
  if (start == null || end == null) return null;
  return (end - start) * 100;
}

export function decorate(points) {
  return {
    ...points,
    stomion: mid(points.lipUpper, points.lipLower),
    pupils: mid(points.irisL, points.irisR),
    browL: mid(points.browOuterL, points.browInnerL),
    browR: mid(points.browOuterR, points.browInnerR),
    glabella: points.glabella || mid(points.browInnerL, points.browInnerR),
  };
}

function pairHoriz(frame, a, b) {
  const A = xy(frame, a);
  const B = xy(frame, b);
  const mean = (Math.abs(A.u) + Math.abs(B.u)) / 2;
  if (!mean) return null;
  return (Math.abs(Math.abs(A.u) - Math.abs(B.u)) / mean) * 100;
}

function pairVert(frame, a, b) {
  const A = xy(frame, a);
  const B = xy(frame, b);
  return Math.abs(A.v - B.v);
}

export const METRICS = [
  {
    id: "face-wh",
    group: "Пропорции",
    name: "Ширина к высоте",
    kind: "ratio",
    points: ["zygionL", "zygionR", "trichion", "forehead", "menton"],
    lines: [["zygionL", "zygionR"], ["forehead", "menton"]],
    segments(p) {
      return [["zygionL", "zygionR"], [p.trichion ? "trichion" : "forehead", "menton"]];
    },
    about: "Крайние боковые точки сетки, делённые на высоту от линии роста до подбородка. Если линия роста не нашлась, верх берётся с границы сетки на лбу.",
    read: (p) => ratio(dist(p.zygionL, p.zygionR), faceSpan(p)),
  },
  {
    id: "temple-w",
    group: "Пропорции",
    name: "Виски к ширине лица",
    kind: "ratio",
    points: ["templeL", "templeR", "zygionL", "zygionR"],
    lines: [["templeL", "templeR"], ["zygionL", "zygionR"]],
    about: "Ширина по боковым точкам сетки у лба к ширине лица. Это контур сетки, не костный висок.",
    read: (p) => ratio(dist(p.templeL, p.templeR), dist(p.zygionL, p.zygionR)),
  },
  {
    id: "jaw-w",
    group: "Пропорции",
    name: "Челюсть к ширине лица",
    kind: "ratio",
    points: ["gonionL", "gonionR", "zygionL", "zygionR"],
    lines: [["gonionL", "gonionR"], ["zygionL", "zygionR"]],
    about: "Ширина по точкам угла челюсти на контуре к ширине лица.",
    read: (p) => ratio(dist(p.gonionL, p.gonionR), dist(p.zygionL, p.zygionR)),
  },
  {
    id: "wide-h",
    group: "Пропорции",
    name: "Высота самой широкой точки",
    kind: "pct",
    points: ["zygionL", "zygionR", "trichion", "forehead", "menton"],
    lines: [["zygionL", "zygionR"], ["forehead", "menton"]],
    segments(p) {
      return [["zygionL", "zygionR"], [p.trichion ? "trichion" : "forehead", "menton"]];
    },
    about: "Где по высоте лица лежит самая широкая пара точек. 0% — линия роста, если она стоит, иначе верх сетки на лбу. 100% — подбородок.",
    read: (p) => {
      const top = topOfFace(p);
      const left = along(top, p.menton, p.zygionL);
      const right = along(top, p.menton, p.zygionR);
      if (left == null || right == null) return null;
      return ((left + right) / 2) * 100;
    },
  },
  {
    id: "third-upper",
    group: "Пропорции",
    name: "Верхняя треть",
    kind: "pct",
    points: ["trichion", "glabella", "subnasale", "menton"],
    lines: [["trichion", "glabella"]],
    guides: ["span"],
    about: "Линия роста — точка между бровями, доля высоты до подбородка. Граница не назион: назион ниже, там уже начинается нос. Линию роста можно перетащить. Если её нет — выбери эту строку и кликни по волосам.",
    read: (p) => thirdBetween(p, p.trichion, p.glabella),
  },
  {
    id: "third-mid",
    group: "Пропорции",
    name: "Средняя треть",
    kind: "pct",
    points: ["trichion", "glabella", "subnasale", "menton"],
    lines: [["glabella", "subnasale"]],
    guides: ["span"],
    about: "Точка между бровями — субназале, доля той же высоты. Эту точку можно сдвинуть: пересчитаются все три трети.",
    read: (p) => thirdBetween(p, p.glabella, p.subnasale),
  },
  {
    id: "third-low",
    group: "Пропорции",
    name: "Нижняя треть",
    kind: "pct",
    points: ["trichion", "glabella", "subnasale", "menton"],
    lines: [["subnasale", "menton"]],
    guides: ["span"],
    about: "Субназале — подбородок, доля той же высоты. Верхняя, средняя и нижняя трети в сумме дают 100%.",
    read: (p) => thirdBetween(p, p.subnasale, p.menton),
  },
  {
    id: "lower-mid",
    group: "Пропорции",
    name: "Нижняя треть к средней",
    kind: "ratio",
    points: ["glabella", "subnasale", "menton"],
    lines: [["glabella", "subnasale"], ["subnasale", "menton"]],
    about: "Отрезок от точки между бровями до основания носа против отрезка от основания носа до подбородка.",
    read: (p) => ratio(dist(p.subnasale, p.menton), dist(p.glabella, p.subnasale)),
  },
  {
    id: "brow-face",
    group: "Пропорции",
    name: "Высота от бровей к ширине",
    kind: "ratio",
    points: ["glabella", "menton", "zygionL", "zygionR"],
    lines: [["glabella", "menton"], ["zygionL", "zygionR"]],
    about: "Высота от точки между бровями до подбородка, делённая на ширину лица. Это обычный лицевой индекс без лба и линии роста. Название «общее соотношение лица» у того бота было без формулы, поэтому здесь оно не угадано.",
    read: (p) => ratio(dist(p.glabella, p.menton), dist(p.zygionL, p.zygionR)),
  },
  {
    id: "midface",
    group: "Пропорции",
    name: "Средняя зона",
    kind: "ratio",
    points: ["irisL", "irisR", "lipUpper", "lipLower"],
    lines: [["pupils", "stomion"], ["irisL", "irisR"]],
    about: "Расстояние от линии зрачков до щели рта, делённое на межзрачковое расстояние.",
    read: (p) => ratio(dist(p.pupils, p.stomion), dist(p.irisL, p.irisR)),
  },
  {
    id: "eye-gap",
    group: "Глаза",
    name: "Межкантальное к ширине глаза",
    kind: "ratio",
    points: ["eyeInnerL", "eyeInnerR", "eyeOuterL", "eyeOuterR"],
    lines: [["eyeInnerL", "eyeInnerR"], ["eyeOuterL", "eyeInnerL"], ["eyeInnerR", "eyeOuterR"]],
    about: "Расстояние между внутренними углами глаз, делённое на среднюю ширину глазной щели. В неоклассической схеме здесь около единицы.",
    read: (p) => {
      const eye = (dist(p.eyeOuterL, p.eyeInnerL) + dist(p.eyeInnerR, p.eyeOuterR)) / 2;
      return ratio(dist(p.eyeInnerL, p.eyeInnerR), eye);
    },
  },
  {
    id: "eye-shape-l",
    group: "Глаза",
    name: "Форма глаза, левый",
    kind: "ratio",
    points: ["eyeOuterL", "eyeInnerL", "eyeTopL", "eyeBotL"],
    lines: [["eyeOuterL", "eyeInnerL"], ["eyeTopL", "eyeBotL"]],
    about: "Ширина левой глазной щели, делённая на её высоту.",
    read: (p) => ratio(dist(p.eyeOuterL, p.eyeInnerL), dist(p.eyeTopL, p.eyeBotL)),
  },
  {
    id: "eye-shape-r",
    group: "Глаза",
    name: "Форма глаза, правый",
    kind: "ratio",
    points: ["eyeOuterR", "eyeInnerR", "eyeTopR", "eyeBotR"],
    lines: [["eyeOuterR", "eyeInnerR"], ["eyeTopR", "eyeBotR"]],
    about: "Ширина правой глазной щели, делённая на её высоту.",
    read: (p) => ratio(dist(p.eyeOuterR, p.eyeInnerR), dist(p.eyeTopR, p.eyeBotR)),
  },
  {
    id: "ipd-w",
    group: "Глаза",
    name: "Межзрачковое к ширине лица",
    kind: "ratio",
    points: ["irisL", "irisR", "zygionL", "zygionR"],
    lines: [["irisL", "irisR"], ["zygionL", "zygionR"]],
    about: "Расстояние между зрачками, делённое на ширину лица.",
    read: (p) => ratio(dist(p.irisL, p.irisR), dist(p.zygionL, p.zygionR)),
  },
  {
    id: "intercanthal-face",
    group: "Глаза",
    name: "Межкантальное к ширине лица",
    kind: "ratio",
    points: ["eyeInnerL", "eyeInnerR", "zygionL", "zygionR"],
    lines: [["eyeInnerL", "eyeInnerR"], ["zygionL", "zygionR"]],
    about: "Расстояние между внутренними углами глаз, делённое на ширину лица. Это не тест «между глаз помещается ещё один глаз»: там делитель — ширина глаза.",
    read: (p) => ratio(dist(p.eyeInnerL, p.eyeInnerR), dist(p.zygionL, p.zygionR)),
  },
  {
    id: "eye-face",
    group: "Глаза",
    name: "Ширина глаза к ширине лица",
    kind: "ratio",
    points: ["eyeOuterL", "eyeInnerL", "eyeOuterR", "eyeInnerR", "zygionL", "zygionR"],
    lines: [["eyeOuterL", "eyeInnerL"], ["eyeInnerR", "eyeOuterR"], ["zygionL", "zygionR"]],
    about: "Средняя ширина глазной щели, делённая на ширину лица. В каноне пяти долей лица здесь одна пятая.",
    read: (p) => {
      const eye = (dist(p.eyeOuterL, p.eyeInnerL) + dist(p.eyeInnerR, p.eyeOuterR)) / 2;
      return ratio(eye, dist(p.zygionL, p.zygionR));
    },
  },
  {
    id: "canthal-l",
    group: "Глаза",
    name: "Кантальный наклон, левый",
    kind: "deg",
    signed: true,
    points: ["eyeInnerL", "eyeOuterL"],
    lines: [["eyeInnerL", "eyeOuterL"]],
    about: "Наклон левой глазной щели к горизонтали лица. Плюс — наружный угол выше внутреннего. Горизонталь перпендикулярна линии назион–подбородок, это не рамка фото.",
    read: (p, f) => tilt(f, p.eyeInnerL, p.eyeOuterL),
  },
  {
    id: "canthal-r",
    group: "Глаза",
    name: "Кантальный наклон, правый",
    kind: "deg",
    signed: true,
    points: ["eyeInnerR", "eyeOuterR"],
    lines: [["eyeInnerR", "eyeOuterR"]],
    about: "Наклон правой глазной щели к горизонтали лица. Плюс — наружный угол выше внутреннего.",
    read: (p, f) => tilt(f, p.eyeInnerR, p.eyeOuterR),
  },
  {
    id: "brow-gap-l",
    group: "Брови",
    name: "Зазор бровь–веко, левый",
    kind: "ratio",
    points: ["browOuterL", "browInnerL", "eyeTopL"],
    lines: [["browOuterL", "browInnerL"], ["browL", "eyeTopL"]],
    about: "Зазор от середины левой брови до верхнего века, в высотах этой глазной щели. Меньше — бровь ближе к глазу.",
    read: (p, f) => {
      const gap = xy(f, p.eyeTopL).v - xy(f, p.browL).v;
      return ratio(gap, dist(p.eyeTopL, p.eyeBotL));
    },
  },
  {
    id: "brow-gap-r",
    group: "Брови",
    name: "Зазор бровь–веко, правый",
    kind: "ratio",
    points: ["browOuterR", "browInnerR", "eyeTopR"],
    lines: [["browOuterR", "browInnerR"], ["browR", "eyeTopR"]],
    about: "Зазор от середины правой брови до верхнего века, в высотах этой глазной щели.",
    read: (p, f) => {
      const gap = xy(f, p.eyeTopR).v - xy(f, p.browR).v;
      return ratio(gap, dist(p.eyeTopR, p.eyeBotR));
    },
  },
  {
    id: "brow-tilt-l",
    group: "Брови",
    name: "Наклон брови, левый",
    kind: "deg",
    signed: true,
    points: ["browInnerL", "browOuterL"],
    lines: [["browInnerL", "browOuterL"]],
    about: "Наклон левой брови к горизонтали лица. Плюс — хвост выше головки.",
    read: (p, f) => tilt(f, p.browInnerL, p.browOuterL),
  },
  {
    id: "brow-tilt-r",
    group: "Брови",
    name: "Наклон брови, правый",
    kind: "deg",
    signed: true,
    points: ["browInnerR", "browOuterR"],
    lines: [["browInnerR", "browOuterR"]],
    about: "Наклон правой брови к горизонтали лица. Плюс — хвост выше головки.",
    read: (p, f) => tilt(f, p.browInnerR, p.browOuterR),
  },
  {
    id: "brow-len-l",
    group: "Брови",
    name: "Длина брови к глазу, левая",
    kind: "ratio",
    points: ["browOuterL", "browInnerL", "eyeOuterL", "eyeInnerL"],
    lines: [["browOuterL", "browInnerL"], ["eyeOuterL", "eyeInnerL"]],
    about: "Длина левой брови от головки до хвоста, делённая на ширину левой глазной щели.",
    read: (p) => ratio(dist(p.browOuterL, p.browInnerL), dist(p.eyeOuterL, p.eyeInnerL)),
  },
  {
    id: "brow-len-r",
    group: "Брови",
    name: "Длина брови к глазу, правая",
    kind: "ratio",
    points: ["browOuterR", "browInnerR", "eyeOuterR", "eyeInnerR"],
    lines: [["browOuterR", "browInnerR"], ["eyeOuterR", "eyeInnerR"]],
    about: "Длина правой брови от головки до хвоста, делённая на ширину правой глазной щели.",
    read: (p) => ratio(dist(p.browOuterR, p.browInnerR), dist(p.eyeOuterR, p.eyeInnerR)),
  },
  {
    id: "bridge-len",
    group: "Нос",
    name: "Длина спинки",
    kind: "pct",
    points: ["nasion", "pronasale", "forehead", "menton"],
    lines: [["nasion", "pronasale"]],
    about: "Назион–кончик носа, в процентах высоты от линии роста до подбородка.",
    read: (p) => percent(dist(p.nasion, p.pronasale), faceSpan(p)),
  },
  {
    id: "nose-canthi",
    group: "Нос",
    name: "Ширина носа к межкантальному",
    kind: "ratio",
    points: ["alarL", "alarR", "eyeInnerL", "eyeInnerR"],
    lines: [["alarL", "alarR"], ["eyeInnerL", "eyeInnerR"]],
    about: "Ширина по крыльям носа, делённая на расстояние между внутренними углами глаз.",
    read: (p) => ratio(dist(p.alarL, p.alarR), dist(p.eyeInnerL, p.eyeInnerR)),
  },
  {
    id: "nose-face",
    group: "Нос",
    name: "Ширина носа к ширине лица",
    kind: "ratio",
    points: ["alarL", "alarR", "zygionL", "zygionR"],
    lines: [["alarL", "alarR"], ["zygionL", "zygionR"]],
    about: "Ширина по крыльям носа, делённая на ширину лица. В каноне пяти долей средняя доля — одна пятая.",
    read: (p) => ratio(dist(p.alarL, p.alarR), dist(p.zygionL, p.zygionR)),
  },
  {
    id: "nostril-l",
    group: "Нос",
    name: "Наклон ноздри, левый",
    kind: "deg",
    points: ["nostrilTopL", "nostrilLowL"],
    lines: [["nostrilTopL", "nostrilLowL"]],
    about: "Наклон левой ноздри к горизонтали лица: верхний край отверстия к нижнебоковому, как их ставит сетка. 0° — горизонтально. Если точка не на ноздре, сдвинь её.",
    read: (p, f) => incline(f, p.nostrilTopL, p.nostrilLowL),
  },
  {
    id: "nostril-r",
    group: "Нос",
    name: "Наклон ноздри, правый",
    kind: "deg",
    points: ["nostrilTopR", "nostrilLowR"],
    lines: [["nostrilTopR", "nostrilLowR"]],
    about: "Наклон правой ноздри к горизонтали лица. 0° — горизонтально.",
    read: (p, f) => incline(f, p.nostrilTopR, p.nostrilLowR),
  },
  {
    id: "tip-shift",
    group: "Нос",
    name: "Смещение кончика",
    kind: "pct",
    signed: true,
    points: ["pronasale", "nasion", "menton"],
    lines: [["nasion", "pronasale"]],
    guides: ["midline"],
    about: "Горизонтальный сдвиг кончика от средней линии, в процентах ширины лица. Плюс — вправо на фото.",
    read: (p, f) => percent(xy(f, p.pronasale).u, dist(p.zygionL, p.zygionR)),
  },
  {
    id: "lip-ratio",
    group: "Губы и подбородок",
    name: "Нижняя губа к верхней",
    kind: "ratio",
    points: ["cupid", "lipUpper", "lipLower", "lipBot"],
    lines: [["cupid", "lipUpper"], ["lipLower", "lipBot"]],
    about: "Высота нижней красной каймы к высоте верхней. Верхняя — от лука Купидона до смыкания, нижняя — от смыкания до нижнего края.",
    read: (p) => ratio(dist(p.lipLower, p.lipBot), dist(p.cupid, p.lipUpper)),
  },
  {
    id: "cupid",
    group: "Губы и подбородок",
    name: "Глубина лука Купидона",
    kind: "pct",
    signed: true,
    points: ["cupid", "cupidL", "cupidR", "mouthL", "mouthR"],
    lines: [["cupidL", "cupid"], ["cupid", "cupidR"], ["mouthL", "mouthR"]],
    about: "Насколько ямка лука ниже его пиков, в процентах ширины рта. Плюс — ямка ниже пиков.",
    read: (p, f) => {
      const dip = xy(f, p.cupid).v - (xy(f, p.cupidL).v + xy(f, p.cupidR).v) / 2;
      return percent(dip, dist(p.mouthL, p.mouthR));
    },
  },
  {
    id: "mouth-nose",
    group: "Губы и подбородок",
    name: "Ширина рта к ширине носа",
    kind: "ratio",
    points: ["mouthL", "mouthR", "alarL", "alarR"],
    lines: [["mouthL", "mouthR"], ["alarL", "alarR"]],
    about: "Расстояние между углами рта, делённое на ширину крыльев носа.",
    read: (p) => ratio(dist(p.mouthL, p.mouthR), dist(p.alarL, p.alarR)),
  },
  {
    id: "mouth-face",
    group: "Губы и подбородок",
    name: "Ширина рта к ширине лица",
    kind: "ratio",
    points: ["mouthL", "mouthR", "zygionL", "zygionR"],
    lines: [["mouthL", "mouthR"], ["zygionL", "zygionR"]],
    about: "Расстояние между углами рта, делённое на ширину лица.",
    read: (p) => ratio(dist(p.mouthL, p.mouthR), dist(p.zygionL, p.zygionR)),
  },
  {
    id: "chin-philtrum",
    group: "Губы и подбородок",
    name: "Подбородок к филтруму",
    kind: "ratio",
    points: ["subnasale", "lipUpper", "lipLower", "menton"],
    lines: [["subnasale", "stomion"], ["stomion", "menton"]],
    about: "Отрезок от щели рта до подбородка, делённый на отрезок от субназале до щели рта.",
    read: (p) => ratio(dist(p.stomion, p.menton), dist(p.subnasale, p.stomion)),
  },
  {
    id: "pupil-mouth",
    group: "Губы и подбородок",
    name: "Зрачки к рту",
    kind: "pct",
    points: ["irisL", "irisR", "lipUpper", "lipLower", "forehead", "menton"],
    lines: [["pupils", "stomion"]],
    about: "Расстояние от линии зрачков до щели рта, в процентах высоты от линии роста до подбородка.",
    read: (p) => percent(dist(p.pupils, p.stomion), faceSpan(p)),
  },
  {
    id: "corners",
    group: "Губы и подбородок",
    name: "Сдвиг уголков",
    kind: "pct",
    signed: true,
    points: ["mouthL", "mouthR", "lipUpper", "lipLower"],
    lines: [["mouthL", "mouthR"]],
    about: "Сдвиг уголков относительно щели рта, в процентах ширины рта. Плюс — уголки выше.",
    read: (p, f) => {
      const lift = xy(f, p.stomion).v - (xy(f, p.mouthL).v + xy(f, p.mouthR).v) / 2;
      return percent(lift, dist(p.mouthL, p.mouthR));
    },
  },
  {
    id: "jaw-angle-l",
    group: "Челюсть",
    name: "Угол контура, левый",
    kind: "deg",
    points: ["zygionL", "gonionL", "menton"],
    lines: [["zygionL", "gonionL"], ["gonionL", "menton"]],
    about: "Угол в левой точке угла челюсти: от боковой точки сетки к углу и от угла к подбородку. Контур сетки, не костный гониальный угол.",
    read: (p) => angleAt(p.zygionL, p.gonionL, p.menton),
  },
  {
    id: "jaw-angle-r",
    group: "Челюсть",
    name: "Угол контура, правый",
    kind: "deg",
    points: ["zygionR", "gonionR", "menton"],
    lines: [["zygionR", "gonionR"], ["gonionR", "menton"]],
    about: "Угол в правой точке угла челюсти. Контур сетки, не костный гониальный угол.",
    read: (p) => angleAt(p.zygionR, p.gonionR, p.menton),
  },
  {
    id: "jaw-slope-l",
    group: "Челюсть",
    name: "Наклон линии челюсти, левый",
    kind: "deg",
    points: ["gonionL", "menton"],
    lines: [["gonionL", "menton"]],
    about: "Наклон левой линии «угол челюсти — подбородок» к горизонтали лица. 0° — горизонтально, 90° — отвесно.",
    read: (p, f) => incline(f, p.gonionL, p.menton),
  },
  {
    id: "jaw-slope-r",
    group: "Челюсть",
    name: "Наклон линии челюсти, правый",
    kind: "deg",
    points: ["gonionR", "menton"],
    lines: [["gonionR", "menton"]],
    about: "Наклон правой линии «угол челюсти — подбородок» к горизонтали лица.",
    read: (p, f) => incline(f, p.gonionR, p.menton),
  },
  {
    id: "ala-jaw",
    group: "Челюсть",
    name: "Крыло–рот против наклона челюсти",
    kind: "deg",
    points: ["alarL", "mouthL", "gonionL", "alarR", "mouthR", "gonionR", "menton"],
    lines: [["alarL", "mouthL"], ["gonionL", "menton"], ["alarR", "mouthR"], ["gonionR", "menton"]],
    about: "На сколько наклон от крыла носа к углу рта отличается от наклона линии челюсти, среднее двух сторон. Так я читаю их «отклонение IAA и JFA». Формулы того бота под рукой нет.",
    read: (p, f) => {
      const left = Math.abs(incline(f, p.alarL, p.mouthL) - incline(f, p.gonionL, p.menton));
      const right = Math.abs(incline(f, p.alarR, p.mouthR) - incline(f, p.gonionR, p.menton));
      return (left + right) / 2;
    },
  },
];

const PAIRS = [
  ["sym-jaw", "Челюсть", "gonionL", "gonionR", "Разнобой левой и правой точек угла челюсти по горизонтали лица."],
  ["sym-eyes", "Глаза", "irisL", "irisR", "Разнобой зрачков по горизонтали лица."],
  ["sym-brow", "Брови", "browL", "browR", "Разнобой середин бровей по горизонтали лица."],
  ["sym-cheek", "Скулы", "zygionL", "zygionR", "Разнобой крайних боковых точек по горизонтали лица. Это ширина сетки, не костная скула."],
  ["sym-alar", "Крылья носа", "alarL", "alarR", "Разнобой крыльев носа по горизонтали лица."],
  ["sym-mouth", "Рот", "mouthL", "mouthR", "Разнобой углов рта по горизонтали лица."],
  ["sym-temple", "Виски", "templeL", "templeR", "Разнобой боковых точек у лба по горизонтали лица."],
];

for (const [id, name, a, b, about] of PAIRS) {
  METRICS.push({
    id,
    group: "Симметрия",
    name,
    kind: "pct",
    points: [a, b, "nasion", "menton"],
    lines: [],
    guides: ["midline"],
    about: about + " Разница расстояний до средней линии, в процентах их среднего. 0% — на равном удалении. Средняя линия — назион–подбородок.",
    read: (p, f) => pairHoriz(f, p[a], p[b]),
  });
}

METRICS.push({
  id: "sym-bridge",
  group: "Симметрия",
  name: "Спинка носа",
  kind: "pct",
  points: ["bridgeUpper", "bridgeMid", "bridgeLower", "pronasale", "nasion", "menton"],
  lines: [["nasion", "pronasale"]],
  guides: ["midline"],
  about: "Средний уход точек спинки и кончика от средней линии, в процентах полуширины лица.",
  read: (p, f) => {
    const ids = ["bridgeUpper", "bridgeMid", "bridgeLower", "pronasale"];
    const mean = ids.reduce((sum, id) => sum + Math.abs(xy(f, p[id]).u), 0) / ids.length;
    const half = dist(p.zygionL, p.zygionR) / 2;
    return percent(mean, half);
  },
});

METRICS.push({
  id: "sym-vert",
  group: "Симметрия",
  name: "Вертикальный разнобой",
  kind: "pct",
  points: ["gonionL", "gonionR", "irisL", "irisR", "mouthL", "mouthR", "forehead", "menton"],
  lines: [],
  guides: ["midline"],
    about: "Средний вертикальный разнобой парных точек челюсти, зрачков, бровей, краёв лица, крыльев, рта и висков. В процентах высоты от линии роста до подбородка. Миллиметров нет: масштаба в кадре нет.",
    read: (p, f) => {
    const pairs = [
      ["gonionL", "gonionR"],
      ["irisL", "irisR"],
      ["browL", "browR"],
      ["zygionL", "zygionR"],
      ["alarL", "alarR"],
      ["mouthL", "mouthR"],
      ["templeL", "templeR"],
    ];
    const mean = pairs.reduce((sum, [a, b]) => sum + pairVert(f, p[a], p[b]), 0) / pairs.length;
    return percent(mean, faceSpan(p));
  },
});

export const GROUPS = [];
for (const metric of METRICS) {
  let group = GROUPS.find((g) => g.name === metric.group);
  if (!group) {
    group = { name: metric.group, items: [] };
    GROUPS.push(group);
  }
  group.items.push(metric);
}

export function measure(points) {
  if (!points || POINTS.some((spec) => !points[spec.id])) return null;
  const p = decorate(points);
  const frame = faceFrame(p);
  const out = {};
  for (const metric of METRICS) {
    let value = null;
    try {
      value = metric.read(p, frame);
    } catch {
      value = null;
    }
    out[metric.id] = Number.isFinite(value) ? value : null;
  }
  return out;
}

export function poseOf(points) {
  const roll = (Math.atan2(points.irisR.y - points.irisL.y, points.irisR.x - points.irisL.x) * 180) / Math.PI;
  const frame = faceFrame(points);
  const left = Math.abs(xy(frame, points.zygionL).u);
  const right = Math.abs(xy(frame, points.zygionR).u);
  return { roll, half: right ? left / right : null };
}
