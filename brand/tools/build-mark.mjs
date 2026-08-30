#!/usr/bin/env node
// Builds the Ligga fritt brand assets from the Illustrator export.
//
//   node brand/tools/build-mark.mjs
//
// Inputs : brand/liggafritt-source.svg   (Illustrator export, do not hand-edit)
// Outputs: brand/mark.svg                optimised vector — the editable master,
//                                        NOT loaded by the page
//          brand/mark.webp               what the header shows
//          brand/favicon-{16,32,48}.png, brand/apple-touch-icon.png
//
// Square crop around the face, in source viewBox units. Measured, not eyeballed:
// rendering the figure with the halftone stripped and reading the silhouette
// width per row puts the crown at y=92, the widest point of the beard at y~420,
// the neck at its narrowest at y~620, and the shoulders flaring from y~668. So
// the head occupies y 92..620, x 456..902 at its widest, and the crop below
// frames it with a little air above and a collar's worth below.
//
// This does cut the halo, which a full-bust framing would not. That is the
// trade: at 40px in the header a whole bust leaves the face about fifteen
// pixels tall, and a decisive face crop reads as intended where a nearly
// contained halo reads as an accident.
//
// The mark is served as raster. It is a traced portrait with ~500 paths, so the
// vector stays large however hard it is squeezed, while a 160px webp covering
// 3x DPR at the header's 45px is a few kilobytes. At 16-48px a favicon gains
// nothing from being vector.
//
// Requires: npx (svgo), inkscape, cwebp.

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync, execSync } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '..');
const src = path.join(root, 'liggafritt-source.svg');
const tmp = fs.mkdtempSync('/tmp/lfmark-');
const t = (f) => path.join(tmp, f);
const out = (f) => path.join(root, f);

const svgo = (input, output, config) => {
  fs.writeFileSync(t('svgo.config.mjs'), config);
  execSync(`npx -y svgo@3 -i "${input}" -o "${output}" --config "${t('svgo.config.mjs')}"`,
    { stdio: ['ignore', 'ignore', 'inherit'] });
};

const PASS1 = `export default {
  multipass: true,
  plugins: [{ name: 'preset-default', params: { overrides: {
    removeViewBox: false,
    convertPathData: { floatPrecision: 1, transformPrecision: 2 },
    cleanupNumericValues: { floatPrecision: 1 },
    mergePaths: { force: true },
  }}}],
};`;

const PASS2 = `export default {
  multipass: true,
  plugins: [{ name: 'preset-default', params: { overrides: {
    removeViewBox: false,
    inlineStyles: false,
    mergePaths: { force: true },
  }}}],
};`;

/** CIELAB ΔE, so "close enough to merge" is a measurement rather than a guess. */
function deltaE(a, b) {
  const lab = (hex) => {
    const [r, g, bl] = [1, 3, 5]
      .map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
      .map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    const X = r * 0.4124 + g * 0.3576 + bl * 0.1805;
    const Y = r * 0.2126 + g * 0.7152 + bl * 0.0722;
    const Z = r * 0.0193 + g * 0.1192 + bl * 0.9505;
    const f = (v) => (v > 0.008856 ? Math.cbrt(v) : 7.787 * v + 16 / 116);
    const [fx, fy, fz] = [f(X / 0.95047), f(Y), f(Z / 1.08883)];
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
  };
  const [A, B] = [lab(a), lab(b)];
  return Math.hypot(A[0] - B[0], A[1] - B[1], A[2] - B[2]);
}

/**
 * Collapse colour pairs below the perceptual threshold, then hoist every fill
 * into a one-letter class. Deliberate tonal steps — halftone shading, a ramp —
 * sit well above ΔE 2 and are left alone.
 */
function classifyFills(svg) {
  const counts = {};
  for (const m of svg.matchAll(/fill="(#[0-9a-fA-F]{6})"/g)) counts[m[1]] = (counts[m[1]] || 0) + 1;
  const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1]);

  // Fold each colour into the most common one it is indistinguishable from.
  //
  // Compare only against colours that are themselves canonical. Matching against
  // every key instead lets merges chain — A absorbs B, then C matches B and is
  // redirected to A, which it may be nowhere near. On artwork with a handful of
  // flat colours that never shows; on this one, with 900-odd shades, it produced
  // shifts of ΔE 6 and worse. Every colour now lands within the threshold of the
  // colour it actually becomes.
  const canonicals = [];
  const merged = [];
  for (const [colour] of ranked) {
    const hit = canonicals.find((c) => deltaE(c, colour) < MERGE_THRESHOLD);
    if (hit) merged.push([colour, hit]);
    else canonicals.push(colour);
  }
  for (const [from, to] of merged) svg = svg.replaceAll(`fill="${from}"`, `fill="${to}"`);

  const worst = merged.reduce((m, [f, t]) => Math.max(m, deltaE(f, t)), 0);
  console.log(`  merged ${merged.length} colours, worst shift ΔE ${worst.toFixed(2)} (threshold ${MERGE_THRESHOLD})`);

  const finals = {};
  for (const m of svg.matchAll(/fill="(#[0-9a-fA-F]{6})"/g)) finals[m[1]] = (finals[m[1]] || 0) + 1;
  const order = Object.entries(finals).sort((a, b) => b[1] - a[1]);
  const alpha = 'abcdefghijklmnopqrstuvwxyz';
  const cls = Object.fromEntries(order.map(([c], i) => [c, alpha[i]]));

  svg = svg.replace(/ fill="(#[0-9a-fA-F]{6})"/g, (m, c) => (cls[c] ? ` class="${cls[c]}"` : m));
  const css = order.map(([c]) => `.${cls[c]}{fill:${c}}`).join('');
  console.log(`  palette ${ranked.length} -> ${order.length} colours`);
  return svg.replace(/(<svg[^>]*>)/, `$1<style>${css}</style>`);
}

const MERGE_THRESHOLD = 2;

// x, y, side in source viewBox units.
const CROP = { x: 369, y: 52, w: 620 };

const kb = (f) => `${(fs.statSync(f).size / 1024).toFixed(1)} KB`;

/**
 * Reduce to the crop box: rewrite the viewBox and delete every element whose
 * bounding box falls wholly outside it. Cropping by viewBox alone still ships
 * every path of the body, just clipped.
 */
function crop(svg, { x, y, w }) {
  let n = 0;
  const tagged = svg.replace(/<(path|circle|polygon|rect|ellipse)\b/g, (m, tag) => `<${tag} id="e${n++}"`);
  fs.writeFileSync(t('ids.svg'), tagged);

  const boxes = {};
  for (const line of execFileSync('inkscape', ['--query-all', t('ids.svg')], { maxBuffer: 1 << 28 })
    .toString().split('\n')) {
    const p = line.split(',');
    if (p.length === 5 && /^e\d+$/.test(p[0])) boxes[p[0]] = p.slice(1).map(Number);
  }

  let dropped = 0;
  const out = tagged.replace(/<(?:path|circle|polygon|rect|ellipse)\b[^>]*id="(e\d+)"[^>]*\/>/g, (m, id) => {
    const b = boxes[id];
    if (!b) return m.replace(/ id="e\d+"/, '');
    const [bx, by, bw, bh] = b;
    if (bx + bw < x || bx > x + w || by + bh < y || by > y + w) { dropped++; return ''; }
    return m.replace(/ id="e\d+"/, '');
  });
  console.log(`  dropped ${dropped} off-canvas elements`);
  return out.replace(/viewBox="[^"]*"/, `viewBox="${x} ${y} ${w} ${w}"`);
}

// ── vector master ──────────────────────────────────────────────────────────
console.log('mark…');
// Cull before merging: mergePaths fuses shapes into paths spanning the whole
// figure, and one of those clipping the crop drags everything in with it.
svgo(src, t('p1.svg'), PASS1);
fs.writeFileSync(t('cls.svg'), classifyFills(fs.readFileSync(t('p1.svg'), 'utf8')));
fs.writeFileSync(t('crop.svg'), crop(fs.readFileSync(t('cls.svg'), 'utf8'), CROP));
svgo(t('crop.svg'), t('mark.svg'), PASS2);
fs.writeFileSync(out('mark.svg'),
  fs.readFileSync(t('mark.svg'), 'utf8').replace(/(<svg[^>]*>)/, '$1<title>Ligga fritt</title>'));
console.log(`  brand/mark.svg ${kb(out('mark.svg'))} (source, not served)`);

// ── rasters ────────────────────────────────────────────────────────────────
console.log('icons…');
const png = (size, name) => {
  execFileSync('inkscape', ['--export-type=png', `--export-filename=${out(name)}`,
    '-w', String(size), '-h', String(size), out('mark.svg')], { stdio: 'ignore' });
};

for (const [size, name] of [[16, 'favicon-16.png'], [32, 'favicon-32.png'], [48, 'favicon-48.png'], [180, 'apple-touch-icon.png']]) {
  png(size, name);
  console.log(`  brand/${name} ${kb(out(name))}`);
}

// 160px covers the header's 45px mark at 3x DPR.
png(160, 'mark-160.png');
execFileSync('cwebp', ['-q', '88', '-m', '6', out('mark-160.png'), '-o', out('mark.webp')], { stdio: 'ignore' });
fs.rmSync(out('mark-160.png'), { force: true });
console.log(`  brand/mark.webp ${kb(out('mark.webp'))}`);

fs.rmSync(tmp, { recursive: true, force: true });
