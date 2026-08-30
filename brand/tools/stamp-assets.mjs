#!/usr/bin/env node
// Rewrites the ?v= cache-busting token on every local asset in index.htm to a
// short hash of that file's own contents.
//
//   node brand/tools/stamp-assets.mjs
//
// Why: the token used to be the marketing version, "?v=4.1". It stayed put
// across three commits that changed the CSS, so browsers kept serving a stale
// stylesheet and the layout selector rendered with the old rules. A content
// hash changes exactly when the file changes, and never otherwise.
//
// Run this after touching anything under styles/, script.js or brand/.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const root = path.resolve(import.meta.dirname, '..', '..');
const page = path.join(root, 'index.htm');

const hash = (file) =>
  crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex').slice(0, 8);

let html = fs.readFileSync(page, 'utf8');
const stamped = [];
let missing = 0;

// Matches href/src="some/local/path.ext?v=token"
html = html.replace(/(href|src)="([^":]+?\.(?:css|js|svg|png|webp|jpg))\?v=[^"]*"/g, (all, attr, rel) => {
  const file = path.join(root, rel);
  if (!fs.existsSync(file)) {
    console.warn(`  ! ${rel} does not exist — left as is`);
    missing++;
    return all;
  }
  const h = hash(file);
  stamped.push(`${rel} -> ${h}`);
  return `${attr}="${rel}?v=${h}"`;
});

fs.writeFileSync(page, html);
stamped.forEach((line) => console.log(`  ${line}`));
console.log(`${stamped.length} assets stamped${missing ? `, ${missing} missing` : ''}`);
