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
// No crop step, unlike the Banfinator's mark: this artwork already frames
// itself. Measured from the export, the halftone halo spans x 102..1194,
// y 14..1080 inside a 1254 square, so it clears every edge, and only the
// shoulders leave through the bottom — which is right for a bust.
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

const kb = (f) => `${(fs.statSync(f).size / 1024).toFixed(1)} KB`;

// ── vector master ──────────────────────────────────────────────────────────
console.log('mark…');
svgo(src, t('p1.svg'), PASS1);
fs.writeFileSync(t('cls.svg'), classifyFills(fs.readFileSync(t('p1.svg'), 'utf8')));
svgo(t('cls.svg'), t('mark.svg'), PASS2);
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
