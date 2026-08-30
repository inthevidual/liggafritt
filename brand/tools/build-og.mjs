#!/usr/bin/env node
// Renders brand/tools/og-template.html to brand/og.png (1200x630).
//
//   node brand/tools/build-og.mjs
//
// Needs a Chromium and ImageMagick. Uses $CHROME_PATH, else the first Playwright chromium in
// ~/.cache/ms-playwright, else `chromium`/`google-chrome` on PATH.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '..');
const template = path.join(root, 'tools', 'og-template.html');
const out = path.join(root, 'og.png');

function findChrome() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const cache = path.join(os.homedir(), '.cache', 'ms-playwright');
  if (fs.existsSync(cache)) {
    for (const dir of fs.readdirSync(cache).filter((d) => d.startsWith('chromium-')).sort().reverse()) {
      for (const rel of ['chrome-linux64/chrome', 'chrome-linux/chrome', 'chrome-mac/Chromium.app/Contents/MacOS/Chromium']) {
        const p = path.join(cache, dir, rel);
        if (fs.existsSync(p)) return p;
      }
    }
  }
  for (const c of ['chromium', 'google-chrome', 'chromium-browser']) {
    try { return execFileSync('which', [c]).toString().trim(); } catch { /* keep looking */ }
  }
  throw new Error('No Chromium found. Set CHROME_PATH.');
}

// --window-size counts browser chrome, so a 630-tall window yields a ~545-tall
// viewport and the card's footer falls off. Render tall, then crop the card —
// it is anchored at 0,0 with no body margin, so the crop is exact.
const raw = path.join(os.tmpdir(), `og-raw-${process.pid}.png`);

execFileSync(findChrome(), [
  '--headless', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
  '--force-device-scale-factor=1',
  '--window-size=1200,900',
  '--virtual-time-budget=8000',
  `--screenshot=${raw}`,
  `file://${template}`,
], { stdio: ['ignore', 'ignore', 'ignore'] });

execFileSync('convert', [raw, '-crop', '1200x630+0+0', '+repage', out], { stdio: 'inherit' });
fs.rmSync(raw, { force: true });

const { width, height } = { width: 1200, height: 630 };
const dims = execFileSync('identify', ['-format', '%wx%h', out]).toString();
if (dims !== `${width}x${height}`) throw new Error(`expected ${width}x${height}, got ${dims}`);

console.log(`brand/og.png ${dims} ${(fs.statSync(out).size / 1024).toFixed(1)} KB`);
