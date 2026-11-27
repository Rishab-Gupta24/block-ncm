// Draws assets/overview.svg for the README: sequential steps needed to write one 512-token story.
// Usage: node paper/scripts/make_readme_overview.js
const fs = require('fs');
const path = require('path');

const W = 920, H = 440;
const F = 'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif"';
const INK = '#1f2937', MUTED = '#6b7280';
const X0 = 280;            // left edge of the drawings
const P = 10.6, BW = 8;    // one pass: pitch and bar width (same scale for B1 and Block-NCM)
const out = [];
const t = (x, y, s, size = 13, o = '') =>
  out.push(`<text x="${x}" y="${y}" ${F} font-size="${size}" ${o.includes('fill=') ? '' : `fill="${INK}" `}${o}>${s}</text>`);
const tm = (x, y, s, size = 12, o = '') => t(x, y, s, size, `fill="${MUTED}" ${o}`);
const r = (x, y, w, h, fill, stroke = 'none', rx = 1.5, sw = 1) =>
  out.push(`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`);

out.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`);
r(1, 1, W - 2, H - 2, '#ffffff', '#e5e7eb', 14, 1.5);
t(28, 36, 'Writing one 512-token story (4 chunks of 128 tokens): how many steps must run one after another?', 15, 'font-weight="600"');

// row 1: autoregressive
t(28, 88, 'Autoregressive (AR)', 15, 'font-weight="600"');
tm(28, 107, 'one token per step');
for (let i = 0; i < 92; i++) r(X0 + i * 5.2, 76, 3, 30, '#c4b5fd', 'none', 1);
tm(X0 + 92 * 5.2 + 6, 96, '…');
t(888, 90, '≈512 steps', 15, 'font-weight="700" text-anchor="end"');
tm(888, 108, '6.20 s', 12, 'text-anchor="end"');

// row 2: released NCM, chunk after chunk
t(28, 168, 'NCM, chunk after chunk', 15, 'font-weight="600"');
tm(28, 187, '11 passes × 4 chunks');
for (let i = 0; i < 44; i++) {
  const c = Math.floor(i / 11), x = X0 + i * P + c * 8;
  r(x, 152, BW, 40, c % 2 ? '#d1d5db' : '#e5e7eb', '#9ca3af', 1.5, 0.8);
  if (i % 11 === 5) tm(x + 4, 208, `chunk ${c + 1}`, 11, 'text-anchor="middle"');
}
t(888, 170, '44 passes', 15, 'font-weight="700" text-anchor="end"');
tm(888, 188, '1.62 s', 12, 'text-anchor="end"');

// row 3: Block-NCM
t(28, 262, 'Block-NCM (ours)', 15, 'font-weight="700" fill="#15803d"');
tm(28, 281, 'all chunks in parallel');
tm(28, 298, 'first G levels = shared plan');
const segH = 14, gap = 4, y3 = 240;
for (let i = 0; i < 11; i++) {
  const x = X0 + i * P;
  if (i < 5) r(x, y3, BW, 4 * segH + 3 * gap, '#bfdbfe', '#3b82f6', 1.5, 0.9);
  else for (let c = 0; c < 4; c++) {
    r(x, y3 + c * (segH + gap), BW, segH, '#e5e7eb', '#9ca3af', 1.5, 0.8);
    if (c < 3) r(x, y3 + c * (segH + gap) + segH + 0.5, BW, gap - 1, '#fb923c', 'none', 0.5);
  }
}
for (let c = 0; c < 4; c++) tm(X0 - 8, y3 + c * (segH + gap) + 11, `${c + 1}`, 10, 'text-anchor="end"');
// legend
const LX = X0 + 11 * P + 34;
r(LX, y3 + 2, 14, 11, '#bfdbfe', '#3b82f6', 2, 0.9); tm(LX + 22, y3 + 12, 'plan: first G levels attend across all chunks', 12);
r(LX, y3 + 24, 14, 11, '#e5e7eb', '#9ca3af', 2, 0.8); tm(LX + 22, y3 + 34, 'local levels: each chunk on its own', 12);
r(LX, y3 + 48, 14, 5, '#fb923c', 'none', 1); tm(LX + 22, y3 + 56, 'halo: sees 4 positions of each neighbour', 12);
t(888, 262, '11 passes', 15, 'font-weight="700" text-anchor="end"');
tm(888, 280, '0.38 s', 12, 'text-anchor="end"');
t(888, 298, '16.5× faster than AR', 12, 'font-weight="600" fill="#15803d" text-anchor="end"');

// time axis
out.push(`<line x1="${X0}" y1="348" x2="${X0 + 500}" y2="348" stroke="${MUTED}" stroke-width="1.2"/>`);
out.push(`<path d="M ${X0 + 500} 343 L ${X0 + 510} 348 L ${X0 + 500} 353 Z" fill="${MUTED}"/>`);
tm(X0, 368, 'sequential steps (time) →   one bar = one parallel pass; AR row compressed', 11);
tm(28, 404, 'Batch size 1, one RTX 5050 laptop GPU, TinyStories. Block-NCM: G=5 global levels + halo 4. AR: TinyStories-33M.', 11);
tm(28, 421, 'More chunks keep the same 11 passes: 0.39 s for 256 tokens, 0.38 s for 1,024 tokens.', 11);
out.push('</svg>');
fs.mkdirSync(path.join(__dirname, '../../assets'), { recursive: true });
fs.writeFileSync(path.join(__dirname, '../../assets/overview.svg'), out.join('\n'));
console.log('wrote assets/overview.svg');
