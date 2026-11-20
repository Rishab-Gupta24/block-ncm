// Word version of the Block-NCM paper (same content as paper/blockncm.tex).
//   node paper/scripts/make_docx.js            (needs the npm package "docx"; set NODE_PATH if installed elsewhere)
// Figures: paper/figures/*.png from make_figures.py. Output: paper/blockncm.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel,
  WidthType, BorderStyle, ShadingType, LevelFormat, PageNumber, Footer, PageBreak, TableLayoutType,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const FIG = (n) => path.join(ROOT, "figures", n);
const FONT = "Times New Roman";
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1300; // A4
const CONTENT_W = PAGE_W - 2 * MARGIN;
const BLUE = "1F77B4", RED = "B22222";

// ---------------------------------------------------------------- inline markup: **bold**, *italic*, ^sup^, ~sub~
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    else if (t.startsWith("^")) out.push(new TextRun({ text: t.slice(1, -1), superScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), subScript: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), alignment: opts.align ?? AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, before: opts.before ?? 0, line: 276 }, ...(opts.para || {}) });
// a paragraph that starts with a bold run-in heading, as \paragraph{} in LaTeX
const RP = (head, text) => new Paragraph({ children: [new TextRun({ text: head + " ", bold: true }), ...runs(text)],
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 276 } });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
let listInstance = 0;
const bullets = (items) => items.map((t) => new Paragraph({ children: runs(t), numbering: { reference: "bullets", level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 264 } }));
function numbered(items) {
  const inst = ++listInstance;
  return items.map((t) => new Paragraph({ children: runs(t), numbering: { reference: "numbers", level: 0, instance: inst },
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 264 } }));
}

// ---------------------------------------------------------------- figures and tables
let figN = 0, tabN = 0;
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
function figure(file, widthIn, caption) {
  const { w, h, data } = pngSize(FIG(file));
  const wpx = Math.round(widthIn * 96), hpx = Math.round((wpx * h) / w);
  figN += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data, transformation: { width: wpx, height: hpx },
        altText: { title: `Figure ${figN}`, description: caption.slice(0, 200), name: file } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${figN}: `, bold: true, size: 19 }), ...runs(caption, { size: 19 })] }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function cell(text, width, { header = false, bold = false, align = AlignmentType.CENTER, color } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders: { top: header ? border : none, bottom: header ? border : none, left: none, right: none },
    shading: header ? { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: align, spacing: { after: 0 }, keepNext: true, keepLines: true,
      children: runs(text, { size: 18, bold: header || bold, ...(color ? { color } : {}) }) })],
  });
}
// rows: arrays of strings; a string starting with "!" is bold; widths in DXA summing to CONTENT_W or less
function table(caption, header, rows, widths, notes) {
  tabN += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const mk = (r, isHeader) => new TableRow({ tableHeader: isHeader, cantSplit: true, children: r.map((t, i) => {
    const bold = !isHeader && t.startsWith("!");
    return cell(bold ? t.slice(1) : t, widths[i], { header: isHeader, bold,
      align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER });
  }) });
  const trs = [mk(header, true), ...rows.map((r) => mk(r, false))];
  // bottom rule on the last row
  const out = [
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 160, after: 80 }, keepNext: true,
      children: [new TextRun({ text: `Table ${tabN}: `, bold: true, size: 19 }), ...runs(caption, { size: 19 })] }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
      alignment: AlignmentType.CENTER,
      borders: { top: border, bottom: border, left: none, right: none, insideHorizontal: none, insideVertical: none },
      rows: trs }),
  ];
  if (notes) out.push(new Paragraph({ spacing: { before: 60, after: 200 }, children: runs(notes, { size: 17 }) }));
  else out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  return out;
}
// boxed sample text with a red seam marker written as "||"
function sampleBox(label, color, text) {
  const parts = text.split("||");
  const kids = [new TextRun({ text: label + "   ", bold: true, color, size: 19 })];
  parts.forEach((p, i) => {
    kids.push(new TextRun({ text: p, size: 19 }));
    if (i < parts.length - 1) kids.push(new TextRun({ text: " ‖ ", bold: true, color: RED, size: 19 }));
  });
  const b = { style: BorderStyle.SINGLE, size: 6, color: "888888", space: 4 };
  return new Paragraph({ children: kids, border: { top: b, bottom: b, left: b, right: b }, spacing: { after: 100 },
    indent: { left: 120, right: 120 }, alignment: AlignmentType.LEFT });
}

// ---------------------------------------------------------------- content
const W4 = [3000, 1500, 1500, 1500, 1526]; // 5-column table widths
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: [new TextRun({ text: "Block-NCM: Long Stories in a Constant Number of Parallel Passes", bold: true, size: 34 })] }));
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "Rishab Gupta", size: 24 })] }));
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [new TextRun({ text: "Draft, 20 November 2026", size: 22, italics: true })] }));

add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "Abstract", bold: true, size: 24 })] }));
add(P("Next-Concept Models (NCM) generate a 128-token chunk of text coarse to fine in 11 parallel passes, one per level of a residual code hierarchy. Extending them to longer outputs has two obvious options. Generating chunks one after another multiplies the number of sequential passes by the number of chunks. Letting every level attend across all chunks keeps 11 passes but, as we show, fits the trained length and collapses beyond it. We propose *Block-NCM*: the first G coarse levels form a story plan shared by all chunks, the remaining levels are local to each chunk, and a small *halo* lets local levels see a few positions of each neighbour. Any number of chunks still takes 11 passes. Fine-tuned from the released TinyStories NCM on 512-token stories, Block-NCM writes a story 16.6× faster than a smaller (68.5M-parameter) autoregressive (AR) model at batch size 1 on the same GPU (2.3× at batch 32). At the trained length it is within 0.5–2.4 genPPL of fully global attention, with better NLL. At 1.5× the trained length fully global attention degrades from 18.7 to 31.3 genPPL over three seeds, while Block-NCM with G=3 stays at 21.2. A matched per-chunk test shows the collapse starts in the first chunk and grows with the number of global levels. Block-NCM’s text remains about 2× worse than the AR model’s by genPPL, a gap mostly inherited from base NCM; we leave quality to follow-up work.",
  { para: { indent: { left: 500, right: 500 } } }));

// 1 Introduction
add(H1("1  Introduction"));
add(P("Autoregressive (AR) language models need one sequential forward pass per generated token. Next-Concept Models (NCM) [1] instead encode a chunk of text into a hierarchy of residual vector-quantised codes and generate the hierarchy level by level, all positions of a level in parallel. The released TinyStories model produces 128 tokens in 11 sequential passes and is 10–31× faster than diffusion language models at comparable quality on three benchmarks [1]. Its tokenizer and generator, however, see only 128 tokens at a time."));
add(P("This paper asks how to keep NCM’s small, fixed number of passes when the output is several chunks long. Two simple answers exist (Figure 1):"));
add(bullets([
  "**B1, chunks in sequence:** generate chunk m conditioned on chunks 1..m−1. Quality can be good, but the number of passes grows to 11M for M chunks, giving up NCM’s advantage.",
  "**B2, fully global:** treat the M chunks as one long sequence at every level. Still 11 passes, but every level attends across M× more positions.",
]));
add(P("We find that B2 works at the trained length and *collapses* at longer lengths (Section 5.3). Block-NCM sits between the two: only the coarse levels, which carry topic and characters, attend across chunks."));
add(RP("Contributions.", ""));
add(numbered([
  "**Block-NCM**, a chunked NCM whose coarse levels form a shared plan and whose fine levels are local with a small halo. It takes 11 sequential passes for any number of chunks (Section 3).",
  "**Measured speed**: 16.6× faster than an AR model at batch 1 and 2.3× at batch 32 on the same GPU (Section 5.1).",
  "**A quality–robustness trade-off in G**: more global levels give better text at the trained length and worse text beyond it, with fully global attention at the extreme. A matched per-chunk test locates the failure in the plan itself (Section 5.3).",
  "**A record of what failed**: conditioning corruption is required to fine-tune NCM at all, and four other ideas did not work (Appendix D).",
]));

// 2 Background
add(H1("2  Background: Next-Concept Models"));
add(P("NCM [1] pairs a VQ-VAE tokenizer [4] with a generator. The tokenizer maps 128 BPE tokens to a latent sequence and quantises it with residual vector quantisation [5] over levels of increasing length. In the released TinyStories model, a single position at level 0 is followed by 11 target levels of length (8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128): 641 codes per chunk. Level k refines the reconstruction left by levels below k, so coarse levels hold the gist and fine levels hold token-level detail; the last level cannot be dropped without losing tokens (Appendix C)."));
add(P("The generator is a transformer that predicts all codes of level k in one pass, given the running reconstruction from the coarser levels. Generating a chunk therefore takes 11 passes regardless of its length. During training the generator’s conditioning codes are partly replaced by random codes from the same codebook (*conditioning corruption*, mean rate 0.5), which makes it robust to its own sampling errors [1]. We found this step essential for fine-tuning (Appendix D)."));

// 3 Method
add(H1("3  Block-NCM"));
add(figure("fig_architecture.png", 6.4, "Which positions each level attends to, for M=4 chunks. Each row is one level and one parallel pass. B0 never connects chunks; B2 connects them at every level. Block-NCM connects them only at the first G levels (the plan) and lets local levels see h=4 input positions of each neighbouring chunk (halo). All three take 11 sequential passes for any M; B1 (not shown) would take 11M."));
add(P("A story of M chunks is encoded chunk by chunk with the frozen tokenizer, giving M copies of the 11-level hierarchy. Generation runs level by level as in NCM, with all chunks in the batch, so the number of sequential passes stays at 11. What changes is the attention pattern at each level (Figure 1)."));
add(RP("Plan levels (k ≤ G).", "The codes of all M chunks at level k form one sequence, with a learned chunk embedding added to each position. Every position attends to every chunk. These levels are short (8, 16 and 24 positions per chunk for G=3), so cross-chunk attention is cheap. They decide what each chunk is about."));
add(RP("Local levels (k > G).", "Each chunk is its own sequence and also attends to the plan levels’ keys and values. Those are computed once and cached for all local levels, which halves training time without changing the result (losses equal to within 8×10^−7^). Local levels hold 592 of the 641 codes per chunk for G=3."));
add(RP("Halo.", "A local level of chunk m also attends to the level’s *inputs* at the last h positions of chunk m−1 and the first h positions of chunk m+1. These inputs come from coarser levels that are already generated, so the halo adds no sequential pass. It targets the seams, where a sentence must run on from one chunk into the next. It adds 2h positions per chunk per local level, about 10% compute for h=4."));
add(RP("Special cases.", "G=0 without halo is **B0** (independent chunks). G=11 is **B2** (fully global). We study G ∈ {3, 5, 6} with h ∈ {2, 4, 8}."));
add(RP("Training.", "We fine-tune all generator weights of the released TinyStories NCM. Each step sees M=4 chunks of a story. The loss is the cross-entropy of every level’s codes given the clean codes of the coarser levels, with conditioning corruption on the inputs (rate falling linearly from 0.75 to 0.25 over levels; targets stay clean). Padding codes after the end of a story are masked from all metrics."));

// 4 Setup
add(H1("4  Experimental setup"));
add(RP("Data.", "TinyStories [2], stories longer than 384 BPE tokens (4.7% of the corpus): 125,723 training and 1,228 validation stories, cut to 512 tokens (M=4). Evaluation at M=6 (768 tokens) uses models trained only on M=4."));
add(RP("Models.", "B0, B2, and Block-NCM with G=3 or G=5 and halo h=4, all fine-tuned from the released checkpoint with the same recipe: 1,500 steps of 8 stories, AdamW, lr 3×10^−5^ with 200 warm-up steps and cosine decay, gradient checkpointing. Main models are trained with three seeds (one locally in bf16 on an RTX 5050 laptop GPU, two on Kaggle T4s in fp16)."));
add(RP("Metrics.", ""));
add(bullets([
  "**genPPL**: perplexity of generated text under GPT-2 Large [3], the NCM paper’s protocol; 2 sampling seeds × 256 unconditional samples, temperature 1.0, no top-k/top-p, text cut at the first end-of-text token. Lower is better.",
  "**Seam ratio**: genPPL of 64-token windows across a chunk boundary divided by genPPL of 64-token windows inside a chunk. Real text scores 1.01; an AR model’s samples score 1.04.",
  "**NLL**: teacher-forced loss on all validation stories, padding masked, compared with a paired per-story bootstrap (1,000 resamples).",
]));
add(RP("Reference points.", "The released NCM scores 16.38 ± 0.22 genPPL on single 128-token chunks in our reproduction (paper: 15.41 [1]). As an AR reference we use TinyStories-33M [2], a 68.5M-parameter GPT-Neo trained on TinyStories; at 128 tokens it scores 9.68, close to the NCM paper’s 88.6M AR baseline (8.89)."));

// 5 Results
add(H1("5  Results"));
add(H2("5.1  Speed"));
add(figure("fig_speed.png", 3.6, "Wall-clock time per 512-token story on one RTX 5050 laptop GPU (median of 10 runs after 3 warm-ups). Block-NCM time includes decoding codes to tokens. The AR model uses Hugging Face generate with a KV cache, sampling exactly 517 GPT-Neo tokens (≈512 BPE tokens)."));
add(P("Figure 2 compares Block-NCM (G=5, h=4) with TinyStories-33M generating the same length. At batch 1, Block-NCM takes 0.38 s per story and the AR model 6.24 s: **16.6× faster**. At batch 32 the ratio falls to **2.3×** (0.086 vs 0.201 s per story), because Block-NCM is compute-bound while the AR model’s per-token overhead is shared across the batch. Block-NCM’s network is larger than the AR model (a 546 MB fp32 checkpoint against 274 MB), so the speed comes from making 11 passes instead of about 517, not from a smaller model."));
add(P("Two details matter for a fair comparison. First, the AR side uses unoptimised generate without CUDA graphs or fused kernels, so the batch-1 ratio is an upper bound; AR timings varied between 6.2 and 8.1 s across runs. Second, the tokenizer’s decoder must run with TF32 or fp16 matrix multiplies: in plain fp32 it hits a slow path that takes 4.2 s at batch 32. We run the code lookup in fp32 and the decoder in fp16, which matches the fp32 output on 99.96% of tokens."));
add(RP("At 7B scale (estimate).", "A roofline cost model with both models at 7B parameters on an H100 (Appendix B) predicts 26–27× lower batch-1 latency than an AR model for 2,048-token outputs, with AR becoming cheaper per token from batch 31. The same model puts generating chunks one after another (B1) about 2.3× slower than Block-NCM at batch 1. Block-NCM processes about 5.4 positions per output token, so high-throughput serving favours AR; the case for Block-NCM is low-batch, latency-bound generation. In this model Block-NCM is about 5% slower than B2, because halo positions add queries."));

add(H2("5.2  Quality at the trained length"));
add(table("Main models at the trained length (M=4, 512 tokens) and at M=6 (768 tokens, never trained). Means of three training seeds. NLL: seed 2 (one shared encoding); Δ vs B2 with 95% bootstrap CI.",
  ["", "B0", "G3+h4 (ours)", "G5+h4 (ours)", "B2"],
  [["genPPL, M=4", "24.2", "21.1", "19.2", "!18.7"],
   ["seam ratio, M=4", "1.49", "1.20", "1.14", "!1.12"],
   ["genPPL, M=6", "23.8", "21.2", "!20.6", "31.3"],
   ["seam ratio, M=6", "1.50", "1.19", "1.15", "1.14"],
   ["NLL", "1.6151", "1.5993", "!1.5928", "1.6327"],
   ["ΔNLL vs B2 [95% CI]", "−0.017 [−0.018, −0.016]", "−0.033 [−0.034, −0.032]", "!−0.040 [−0.041, −0.039]", "–"],
   ["NLL, chunk 4", "1.686", "!1.666", "1.667", "1.744"]], [2300, 1700, 1700, 1700, 1300]));
add(P("Table 1 shows the main models at the trained length. **B2 writes the best text at M=4** (18.7 genPPL), with Block-NCM G=5 0.5 behind and G=3 2.4 behind. Seams are close to tied (1.12 vs 1.14), and both are near the 1.04 that AR samples reach without any seams. B0, which never connects chunks, has the worst text and seams (24.2, 1.49) and visibly loses track of its characters (Figure 3)."));
add(P("**B2 has the worst NLL of all models**, worse even than B0, on every seed. Both Block-NCM models beat it with non-overlapping confidence intervals. B2’s loss is concentrated on the fine levels (6–11) and on chunk 4: global attention at the finest levels hurts teacher-forced prediction, plausibly because the released model never saw positions beyond 128 at those levels. Teacher-forced NLL and free generation rank B2 differently, so we report both."));
figN += 1;
add(sampleBox("B0 (G=0)", "555555", "… One day, Sam goes to the park with his mom. He sees a girl with a brown helmet. … “Hey, my my car is faster than yours!” Sam says, pointing to his bike.||Okay, go,”,” Ben said. “I will be careful if you want.” They climbed stairs to the other side of the slide. …"));
add(sampleBox("Block-NCM (G=3, h=4)", BLUE, "… Lila jumps in the puddles. But it is too wet and slippery. She comes off and hits her hand on the ground. “Ow, ow, Billy!” Lila cries, rubbing her arm||. Ben runs to the puddle and hugs her. He lets Sara go away. …"));
add(sampleBox("B2 (G=11)", RED, "… He jumped off and bit her hand. “Ow, ow, ow! That hurts!” Lila cried, poking her finger.||“Ow, that hurts!” Ben said. “You hurt!” Lila ran and to him to the bush. …"));
add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 }, children: [
  new TextRun({ text: `Figure ${figN}: `, bold: true, size: 19 }),
  ...runs("Text on either side of the first chunk boundary (‖), verbatim including errors. Each excerpt is from sample 1 of the model’s evaluation file (same sampling seed and index for all three; samples 0 and 1 are the first two that reach chunk 4). B0 switches to different characters at the seam. Block-NCM continues the sentence across the seam but introduces a new name (Sara); B2 keeps both characters.", { size: 19 })] }));

add(H2("5.3  Beyond the trained length"));
add(figure("fig_tradeoff.png", 6.2, "(a) Whole-story genPPL at the trained length and at 1.5× it, for three training seeds. B2 degrades on every seed (19.0→33.5, 18.4→31.0, 18.7→29.4); Block-NCM G=3 does not move. (b) Matched test: change in the genPPL of chunk 1 when the model is asked for 6 chunks instead of 4. Chunk 1 covers tokens 0–127, so story length cannot explain the change."));
add(P("At M=6, 1.5× the trained length, the ranking flips (Figure 4a). B2 degrades from 18.7 to 31.3 genPPL, replicated on all three training seeds, and to 57.3 at sampling temperature 0.8. Block-NCM G=3 goes from 21.1 to 21.2 and G=5 from 19.2 to 20.6. B0, with no cross-chunk attention, does not degrade either. Among one-seed runs without halo, the same pattern holds: G=6 goes from 19.6 to 30.7 and G=5 from 20.0 to 23.8, while G=3 holds (21.8)."));
add(RP("Ruling out story length.", "Whole-story genPPL at M=6 is confounded: most samples end their story within four chunks, as the training stories do, and models that keep writing are scored on more text. Two checks remove this. First, scoring only samples longer than 4.5 chunks keeps the ranking: G=3 22.5, G=5 30.8, B2 49.8 (one seed). Second, a *matched per-chunk test* compares the same positions under both lengths: chunk m of M=4 samples against chunk m of M=6 samples, each scored by GPT-2 Large given the story before it (Figure 5). B2’s chunks 1–4 are all 58–71% worse when it plans 6 chunks. Its *first* chunk goes from 20.6 to 35.3 genPPL, before any story text exists. The damage therefore comes from the plan, not from errors piling up in a long story. It grows with the number of global levels: −1% for G=0, +6% for G=3, +35% for G=5, and +71% for B2 (Figure 4b)."));
add(figure("fig_chunks.png", 3.6, "Matched per-chunk genPPL: each chunk scored given the story before it, in 4-chunk stories (solid) and 6-chunk stories (dashed), temperature 1.0, one training seed, 430–512 samples per point. † This G=5 run is the weakest of four seeds (whole-story M=6 genPPL 25.2 vs 20.0–21.0 for the other three)."));
add(RP("Interpretation.", "Attention that spans all chunks learns the number of chunks it was trained on. When the plan must cover more chunks, every chunk generated under it suffers. Block-NCM confines cross-chunk attention to a few short coarse levels and leaves most of the model local, so most of the model never sees the change in length. The choice of G is a trade-off: G=5 is 1.9 genPPL better than G=3 at the trained length, and G=3 is nearly unaffected by length. No model extends gracefully, though: text beyond token 512 scores worse than text within it for all of them."));

add(H2("5.4  Ablations"));
add(table("Ablations at M=4 (one training seed). genPPL and seam ratio; M=6 genPPL in the last column.",
  ["variant", "genPPL", "seams", "M=6"],
  [["B0 (G=0)", "24.1", "1.52", "23.7"], ["G=3, no halo", "22.0", "1.39", "21.8"], ["G=3, h=2", "21.4", "1.22", "21.1"],
   ["G=3, h=4 (local run)", "21.0", "1.19", "21.5"], ["G=3, h=8", "20.4", "1.17", "20.4"], ["G=5, no halo", "20.0", "1.26", "23.8"],
   ["G=5, h=4", "19.1", "1.14", "20.0"], ["G=6, no halo", "19.6", "1.22", "30.7"], ["G=6, h=4", "18.7", "1.12", "23.6"],
   ["B2 (G=11)", "19.0", "1.17", "33.5"], ["G=5, h=4 (local run)", "19.4", "1.17", "25.2"],
   ["G=3, h=4 + plan search (8)", "20.0", "1.21", "–"], ["G=5, h=4 + plan search (8)*", "17.1", "1.16", "–"]],
  [3800, 1600, 1600, 1600],
  "Halo and G sweeps: G=3, h=4 is a local run; h=2, 8 and G=5, 6 with halo ran on Kaggle. Plan search is applied to the local runs. *Entropy 4.36→4.32, length 413→406 tokens."));
add(RP("Halo.", "Without halo, the plan alone closes only 38% of the B0→B2 seam gap (G=3: 1.39). A halo of 4 positions brings the seam ratio to 1.19, 95% of the gap, and gives the best NLL of the one-seed runs. Wider halos help a little more (Table 2). Halo also repairs extrapolation for G=5 (23.8→20.0 at M=6) but not for G=6 (23.6)."));
add(RP("Number of plan levels.", "At M=4, genPPL improves steadily with G; at M=6, G ≥ 6 degrades even with halo. G=3 or G=5 with halo are the useful operating points."));
add(RP("Plan search.", "Sampling 8 plans per story and keeping the one the model scores highest, then running the local levels once, improves genPPL from 21.0 to 20.0 for G=3 and from 19.4 to 17.1 for G=5 (local run; entropy 4.36→4.32, length 413→406 tokens). It costs only extra plan-level passes, about 1.3× the latency for G=3."));
add(RP("Temperature.", "Sampling at temperature 0.8 lowers genPPL by about 23% for every model at M=4 and keeps the ranking. It makes failure beyond the trained length worse: B2 goes to 57.3 at M=6."));

add(H2("5.5  Quality against AR and base NCM"));
add(table("Quality reference points (genPPL, GPT-2 Large, temperature 1.0).", ["", "128 tokens", "512 tokens"],
  [["real TinyStories text", "9.67 [1]", "6.1^a^"], ["AR, TinyStories-33M", "9.68", "9.9^b^"], ["released NCM", "16.4", "14.4^c^"],
   ["Block-NCM G=5 + h=4", "–", "19.2"], ["   + plan search (local run)", "–", "17.1"]],
  [4200, 2200, 2200],
  "^a^Long stories only. ^b^End-of-text blocked to match our length. ^c^Four independent 128-token openings, not one story: seam ratio 1.25, entropy 4.80; each fresh opening scores well under GPT-2 Large."));
add(P("Block-NCM’s text is about 2× worse than the AR model’s by genPPL (19.2 vs 9.9 at 512 tokens; Table 3). Most of this gap is inherited: the released NCM is already 1.7× behind the AR model on single chunks (16.4 vs 9.7), as in the NCM paper’s own comparison (15.41 vs 8.89) [1]. Fine-tuning adds about 4 genPPL per chunk. About 1.6 of that is a change of style: stories longer than 384 tokens never open with “Once upon a time” (0% vs 58% of all stories), and GPT-2 Large scores real long-story openings 1.10× higher than all openings. The remaining ≈2.4 is a cost of our recipe; trained on stories of all lengths, the same recipe reaches 17.0 on single chunks (Appendix D). Part of the AR gap is also style, since TinyStories-33M writes mostly “Once upon a time” openings."));

// 6 Related work
add(H1("6  Related work"));
add(RP("Parallel and non-autoregressive text generation.", "Non-autoregressive translation [6] and iterative mask-predict decoding [7, 8] generate many tokens per pass. Discrete diffusion language models [9, 10] refine all positions over many steps; block diffusion [11] generates blocks in sequence with diffusion inside each block, which corresponds to our B1. NCM [1] generates coarse to fine over a residual code hierarchy [4, 5] in a small fixed number of passes. Block-NCM extends NCM beyond its tokenizer’s 128-token window without adding passes."));
add(RP("Local attention.", "Restricting attention to a window plus a few global positions is standard for long inputs [12]. Our halo is a small window across chunk boundaries at the fine levels; our plan levels play the role of global tokens."));
add(RP("Length generalisation.", "Transformers often fail beyond their training length; position encodings designed for extrapolation [13] or interpolation [14] address this for AR models. We observe the same failure across chunks in NCM and reduce it by keeping most attention local."));
add(RP("Speculative decoding.", "Drafting several tokens cheaply and verifying them with an AR model gives lossless speed-ups [15]. DFlash [16] drafts blocks with a diffusion model conditioned on the target model’s hidden features, reporting over 6× acceleration with greedy decoding and 4.1× at temperature 1. A parallel drafter of this kind is a natural use for Block-NCM."));

// 7 Limitations
add(H1("7  Limitations and future work"));
add(bullets([
  "**Quality.** Block-NCM’s text is about 2× worse than an AR model’s by genPPL, and B2 is 0.5 better at the trained length. Improving quality is the subject of follow-up work: training on stories of all lengths, on mixed numbers of chunks, longer schedules, sequence-level distillation from an AR model [17], test-time selection (plan search already gives −2.3 genPPL), and use as a drafter for an AR verifier.",
  "**Scale.** One dataset, one 128-token base model, 1,500 fine-tuning steps, lengths up to 768 tokens. The original NCM recipe (batch 2,048, 50k updates) does not fit on our 8 GB GPU.",
  "**Evidence.** The long-sample and matched per-chunk tests use one training seed, and for G=5 it is the weakest of four runs. genPPL depends on one judge model.",
  "**Speed only at low batch.** At large batch an AR model is cheaper per token.",
  "**Fixed schedule.** Every chunk gets the same 11 levels. The NCM paper notes that an input-dependent schedule “would adapt compute to example complexity” [1]; our tokenizer probe shows NCM can skip intermediate levels (levels 6→9→11 keep 99.5% of tokens with 39% fewer positions; Appendix C), a direction for faster generation.",
]));

// 8 Conclusion
add(H1("8  Conclusion"));
add(P("Block-NCM generates multi-chunk stories in the same 11 parallel passes that NCM uses for one chunk, 16.6× faster than a smaller AR model at batch 1 on the same GPU. Sharing only the coarse levels across chunks, with a small halo at the seams, nearly matches fully global attention at the trained length and avoids its collapse beyond it. The number of shared levels sets a clear trade-off between quality at the trained length and robustness to length."));

// References
add(H1("References"));
const refs = [
  "Kumbong et al. Next-Concept Modeling: Fast, parallel coarse-to-fine text generation. arXiv:2610.14162, 2026.",
  "R. Eldan and Y. Li. TinyStories: How small can language models be and still speak coherent English? arXiv:2305.07759, 2023.",
  "A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners. OpenAI technical report, 2019.",
  "A. van den Oord, O. Vinyals, and K. Kavukcuoglu. Neural discrete representation learning. NeurIPS, 2017.",
  "N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi. SoundStream: An end-to-end neural audio codec. IEEE/ACM TASLP, 2022.",
  "J. Gu, J. Bradbury, C. Xiong, V. O. K. Li, and R. Socher. Non-autoregressive neural machine translation. ICLR, 2018.",
  "M. Ghazvininejad, O. Levy, Y. Liu, and L. Zettlemoyer. Mask-Predict: Parallel decoding of conditional masked language models. EMNLP, 2019.",
  "H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman. MaskGIT: Masked generative image transformer. CVPR, 2022.",
  "A. Lou, C. Meng, and S. Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. ICML, 2024.",
  "S. S. Sahoo et al. Simple and effective masked diffusion language models. NeurIPS, 2024.",
  "M. Arriola et al. Block diffusion: Interpolating between autoregressive and diffusion language models. ICLR, 2025.",
  "I. Beltagy, M. E. Peters, and A. Cohan. Longformer: The long-document transformer. arXiv:2004.05150, 2020.",
  "O. Press, N. A. Smith, and M. Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. ICLR, 2022.",
  "S. Chen, S. Wong, L. Chen, and Y. Tian. Extending context window of large language models via positional interpolation. arXiv:2306.15595, 2023.",
  "Y. Leviathan, M. Kalman, and Y. Matias. Fast inference from transformers via speculative decoding. ICML, 2023.",
  "J. Chen, Y. Liang, and Z. Liu. DFlash: Block diffusion for flash speculative decoding. ICML, 2026. arXiv:2602.06036.",
  "Y. Kim and A. M. Rush. Sequence-level knowledge distillation. EMNLP, 2016.",
];
refs.forEach((r, i) => add(new Paragraph({ spacing: { after: 60 }, indent: { left: 480, hanging: 480 },
  children: [new TextRun({ text: `[${i + 1}]\t`, size: 19 }), new TextRun({ text: r, size: 19 })],
  tabStops: [{ type: "left", position: 480 }] })));

// ---------------------------------------------------------------- appendices
add(new Paragraph({ children: [new PageBreak()] }));
add(H1("Appendix A  Implementation checks"));
add(bullets([
  "With M=1 the chunked model reproduces the released model’s teacher-forced loss for G=0 and G=11 (1.290056 vs 1.290056).",
  "The tokenizer round trip on real 128-token chunks gives 100% token accuracy.",
  "The per-level training path (one short sequence per level) equals the single dense-mask sequence for M=4 and G ∈ {0, 1, 3, 6, 11} to within 10^−6^; the dense path does not fit in 8 GB.",
  "The KV-cached plan gives the same loss as recomputing the plan at every local level (difference ≤ 8×10^−7^ for G ∈ {0, 1, 3, 6, 11}) and the same generated codes under the same seed; it halves the time of a training step (1.21 vs 2.43 s on a T4).",
  "Halo with M=1 changes nothing. Training inputs equal generation inputs bit for bit at every level.",
]));
add(H1("Appendix B  Serving-cost model"));
add(P("The cost model (paper/scripts/serving_cost.py and method_scenarios.py) is analytic, not measured. It assumes 7B parameters for both an AR model (grouped-query attention with 8 KV groups) and NCM, H100 specifications (989 TFLOP/s, 3.35 TB/s, 80 GB), 40% FLOP and 80% bandwidth utilisation, $2.50 per GPU-hour, and 2,048-token outputs (M=16). It counts dense and attention FLOPs, weight loads and AR KV-cache reads, and ignores the tokenizer’s decoder, kernel-launch overhead and low utilisation on small passes. It assumes equal quality at equal size, which this paper does not show."));
add(table("Cost model, 7B parameters on an H100, 2,048-token outputs.", ["method", "positions per token", "batch-1 latency", "vs AR", "AR cheaper from batch"],
  [["AR (GQA-8)", "1", "10.75 s", "1×", "–"], ["B1, chunks in sequence", "5.00", "0.92 s", "12×", "35"], ["B2, fully global", "5.00", "0.38 s", "28×", "33"],
   ["Block-NCM G=3, h=4", "5.50", "0.41 s", "26×", "31"], ["Block-NCM G=5, h=4", "5.38", "0.40 s", "27×", "31"]],
  [2700, 1600, 1600, 1200, 1900]));
add(P("At batch 1 every NCM variant is limited by loading weights, and B1 is about 2.3× slower than Block-NCM with halo because it makes M times as many passes, each over fewer positions. At large batch NCM is compute-bound: it processes about 5 positions per output token, so an AR model becomes cheaper per token from about batch 31. Without grouped-query attention the AR model’s KV cache (1.07 GB per sequence) caps it below batch 64 on 80 GB, and Block-NCM is cheaper at every feasible batch. Block-NCM and B2 differ little at this size: attention is a small share of FLOPs at d=4096, and the saving from local attention matters only for much longer outputs."));
add(H1("Appendix C  Tokenizer level-skip probe"));
add(P("Can the frozen residual tokenizer reconstruct text when some levels are never produced? We encode 1,024 real 128-token chunks, drop levels, decode, and compare tokens. *Re-quantised*: the residual quantiser is re-run without those levels, so later levels absorb the residual (the best case, for a generator trained on that tokenisation). *Hole*: the normal codes with the skipped levels left out at decode time."));
add(table("Token accuracy and exactly reconstructed chunks when levels are skipped (levels numbered 1–11).",
  ["skipped levels", "positions per chunk", "re-quantised: tokens", "exact chunks", "hole: tokens", "exact chunks"],
  [["none", "641", "99.97%", "96.7%", "99.97%", "96.7%"], ["11 (the last)", "513", "98.26%", "17.9%", "98.26%", "17.9%"],
   ["7", "577", "99.90%", "88.7%", "99.61%", "62.7%"], ["7, 8, 10 (keeps 6→9→11)", "393", "99.52%", "57.0%", "98.41%", "14.4%"],
   ["7–10 (keeps 6→11)", "301", "98.71%", "23.5%", "97.10%", "3.6%"], ["4–10", "177", "91.89%", "0%", "84.81%", "0%"]],
  [2500, 1300, 1450, 1200, 1300, 1200]));
add(P("Intermediate levels can be skipped with little loss when the codes are re-quantised: keeping levels 6, 9 and 11 preserves 99.5% of tokens with 39% fewer positions. The last level cannot be skipped. These are upper bounds with perfect codes; a generator would still have to predict the larger jumps."));

// Appendix D: what failed (same text as sections/failures_and_fixes.tex)
add(H1("Appendix D  What failed and how it was fixed"));
add(P("This appendix records, model by model, every failure we hit, the evidence for it, the diagnosis, and what fixed it. Unresolved failures are reported as such. All runs fine-tune the released TinyStories NCM (11 target levels, 641 codes per 128-token chunk) on M=4 chunks (512 tokens) for 1,500 steps of 8 documents. NLL is measured on all 1,228 validation documents with padding masked. genPPL uses GPT-2 Large, 2 seeds × 256 samples, temperature 1.0. The *boundary ratio* is the genPPL of 64-token windows across a chunk seam divided by that of windows inside a chunk (real text: 1.01)."));
add(H2("D.1  Shared pipeline (all models)"));
add(RP("F1. Fine-tuning destroyed free generation.", "*Evidence.* The first M=4 runs (clean inputs, lr 10^−4^) reached good teacher-forced NLL, but genPPL was 47.8–53.5 for every model. Fine-tuned B0 also scored 47.7 on *single* 128-token chunks, against 16.5 for the released model, with no chunking involved. On story openings, fine-tuned B0 beat the released model on NLL (1.179 vs 1.558, better on 100% of documents) yet generated 3× worse. *Diagnosis.* The released generator is trained with *conditioning corruption*: a fraction of input codes is replaced by random codes from the same codebook, at a rate decreasing linearly over levels (mean 0.5; NCM paper, Sec. 3.4 and App. F.3). Clean fine-tuning makes the model trust its conditioning, so at sampling time its own errors compound. We ruled out the alternatives:"));
add(bullets(["train and generation inputs are bit-identical at every level;", "the tokenizer is unchanged by fine-tuning;",
  "sampling settings match the released evaluation;", "a train-save-load-generate control at lr 10^−12^ gives 16.6."]));
add(P("*Fix.* Uniform code replacement with a linear 0.75→0.25 schedule (targets clean) and lr 3×10^−5^, chosen by a single-chunk gate (Table 6). *Outcome.* Single-chunk genPPL 47.7→20.7. A residual gap of about 4–5 genPPL remained (20.5 plateau vs 16.5). It did not respond to learning rate (10^−5^ to 3×10^−5^), to a 4× larger batch, or to per-level loss weighting (20.9)."));
add(P("*Diagnosis of the residual.* The single-chunk test trained on the openings of stories longer than 384 tokens (our M=4 set, 4.7% of TinyStories). These are a different style of story: none opens with “Once upon a time”, against 58% of all stories and 70% of the released model’s samples. Trained on 60k unfiltered openings instead, the same recipe scores 17.0 (entropy 3.98, as released); adding a KL term that anchors the model to the frozen released model gives 16.1, the released level within noise. The anchor alone, on the long-story openings, gives 19.5. **Mostly resolved:** the residual was largely the data subset, not damage from fine-tuning. The M=4 models must train on long stories, so their absolute genPPL carries this style offset (real long-story openings score 1.10× higher than all openings); the cost is the same for every variant, so comparisons between variants remain fair. The anchor does not carry over to the planned models. The reference sees each chunk alone, without the plan or the earlier chunks. At M=4 it therefore pulls the local levels toward ignoring that context. With weight 3 (G3+halo4), NLL rises 1.736→1.895, chunk-4 NLL 1.78→2.20, genPPL 21.0→23.2 and the boundary ratio 1.19→1.34. G5+halo4 behaves the same way (genPPL 19.1→25.7). Weighting chunks 2–4 at 0.3× reduces the NLL cost but not the generation cost. We keep the unanchored recipe."));
add(table("Single-chunk gate: 300 steps on story openings, genPPL of 256 samples (released model: 16.5).", ["recipe", "genPPL", "entropy"],
  [["lr 10^−4^, clean", "41.0", "3.99"], ["lr 10^−5^, clean", "34.1", "3.98"], ["lr 10^−4^, corruption mean 0.25", "24.4", "3.94"],
   ["lr 10^−4^, corruption mean 0.5 (linear)", "22.2", "3.92"], ["lr 10^−4^, corruption 0.5 (flat)", "21.8", "3.92"],
   ["!lr 3×10^−5^, corruption mean 0.5 (chosen)", "!20.7", "3.92"]], [5000, 1600, 1600]));
add(RP("F2. Padding inflated the agreement between models.", "37% of chunk-4 codes are padding: stories end before token 512. Padding is equally easy for every model and diluted the differences. *Fix:* per-document lengths are saved at encoding time and padding codes are masked in every NLL. A per-document re-encoding check confirms the mask lines up with the stored codes. *Outcome:* the B0→B2 gap grew from 0.015 to 0.020 nats in the clean-recipe runs; all reported NLL is padding-free."));
add(RP("F3. Memory: the training graph did not fit in 8 GB.", "The single dense-mask training sequence (M=4, all levels) exceeds 8 GB. *Fix:* a per-level training path (one short sequence per level, built from the generation layouts). Its loss equals the dense path’s to within 10^−6^ for G ∈ {0, 1, 3, 6, 11}, and gradient checkpointing is used throughout."));
add(RP("F4. Mixed precision.", "On the T4 (no bf16), the quantizer’s convolutions produced NaN in fp16, and training continued silently on NaN loss. *Fix:* the quantizer always runs in fp32; training aborts on any non-finite loss; the pipeline greps every stage log for NaN. Separately, the bf16 quantizer loses precision even on the RTX 5050 (loss 3.116 vs 2.965 in fp32). This may explain why our reproduction of the released model scores 16.38 against the paper’s 15.41; **a test is pending**."));
add(RP("F5. Evaluation running out of memory.", "Generation at M=6 and plan search (8 plans) ran out of memory at batch 64. *Fix:* every evaluation retries at half the batch size. Batch size changes only speed, not the samples’ distribution."));
add(H2("D.2  B0: independent chunks (G=0)"));
add(P("*Failure (by design):* no information crosses chunks. With the fixed recipe it has the worst seams (boundary ratio 1.52) and frequent character-name drift between chunks; genPPL is 24.1. *No fix:* B0 is the lower bound that the plan must beat."));
add(H2("D.3  Ours: Block-NCM (G=3, plan over levels 1–3)"));
add(RP("O1. The plan alone did not fix the seams.", "*Evidence.* Ours has the best NLL of the three baselines (1.7438 vs 1.7486 B0 and 1.7780 B2; −0.0049 nats vs B0, 95% CI [−0.0056, −0.0042]). But it closed only about 40% of the B0→B2 gap in generation: genPPL 22.0 vs 19.0, boundary ratio 1.39 vs 1.17. *Diagnosis.* A level-3 plan carries topic and characters. Seams need token-level continuity, though, and that is decided by the local fine levels, which cannot see the neighbouring chunk. *Fix: halo attention.* Each local level also attends to the inputs at the last h positions of the previous chunk and the first h of the next. These inputs come from coarser levels that are already generated, so this adds no sequential passes. It costs 2h extra positions per chunk per local level (about 10% compute for h=4). *Outcome* (h=4): boundary ratio 1.39→1.19, closing 95% of the B0→B2 gap; best NLL of all runs (1.7361); genPPL 22.0→21.0. Plan search (8 sampled plans, keep the most likely) adds a further gain: 20.0. Wider halos help a little: h=2, 4, 8 give genPPL 21.4, 21.0 and 20.4, with boundary ratios 1.22, 1.19 and 1.17."));
add(RP("O2. Recomputing the plan made training slow.", "The reference implementation recomputes the global plan for every local level. *Fix:* compute the plan’s keys and values once and cache them. This is exact (the losses match), and a training step is 2.0× faster on a T4 (1.21 s vs 2.43 s)."));
add(RP("O3. Generation quality is still below the released model.", "Samples keep a coherent plot across all four chunks, but grammar slips are frequent. That is a consequence of F1 and of model size. Plan-only adapters did not help (below). Unfiltered openings are not usable at M=4, per-level loss weighting had no effect, and a KL anchor to the released model made the M=4 models worse (F1). **Open.**"));
add(H2("D.4  Ours with more global levels (G=5, G=6)"));
add(P("More global levels improve generation at the training length: genPPL 20.0 (G=5) and 19.6 (G=6) vs 19.0 for B2. But they *fail to extrapolate*. At M=6 (trained on M=4), G=6 degrades to 30.7 and G=5 to 23.8, while G=3 holds (21.8). G=6 also loses on the fine levels (level 6 NLL 1.86 vs 1.68). *Fix: add halo.* With halo h=4, G=5 goes from 19.2 at M=4 to 20.6 at M=6 (means of three seeds), the best M=6 score of any model; B2 goes from 18.7 to 31.3. At M=4, B2 leads by about 0.5 genPPL and the seams tie (boundary ratio 1.12 vs 1.14). G=6 with halo generates best at M=4 (18.7) but still degrades at M=6 (23.6). One local retraining of G=5 with halo (local encoding, bf16) reached only 25.2 at M=6 against 20.0–21.0 for the three Kaggle seeds. *Resolution:* G=5 with halo is best at the training length. G=3 with halo (21.1→21.2, three seeds) is the most robust beyond it. G ≥ 6 learns the training length even with halo."));
add(P("*Caveat on the M=6 metric.* Samples are scored up to their first end-of-text token. B0 and G=3 with halo mostly end their stories within four chunks at M=6, while B2 and G=5 more often run on, so part of the M=6 gap measures ending on time. Scoring only samples longer than 4.5 chunks keeps the ranking (G=3+halo 22.5, G=5+halo 30.8, B2 49.8 at temperature 1.0). Text past the trained length is worse than in-length text for every model. A matched test removes the confound. We score chunks 1–4 of M=6 samples against the same chunks of M=4 samples, each conditioned on the story before it (one training seed, 512 samples). B2’s chunk 1 already worsens from 20.6 to 35.3 genPPL (71.4 at temperature 0.8), before anything has been written, so the global plan itself is damaged once it spans more chunks than in training. The damage grows with the number of global levels. For chunk 1 it is none for G=0, +6% for G=3+halo, +35% for G=5+halo (the local run, our weakest G=5 seed) and +71% for B2."));
add(H2("D.5  B2: fully global (G=11)"));
add(P("B2 generates best at M=4 but has the *worst* NLL of all runs (1.7780), losing on levels 6–11 and on chunk 4. More importantly, it **collapses beyond the training length**: genPPL 19.0→33.5 at M=6, and 18.4→31.0 and 18.7→29.4 in two more training seeds, with high sampling variance (sd 2.3 and 1.2). At sampling temperature 0.8 it is worse still (57.3). In the matched per-chunk test above, every one of its first four chunks is 60–70% worse when it plans 6 chunks, so the collapse is not a story-length artifact. Global attention across all chunks fits the training length. This is the baseline’s failure, not ours, and it motivates Block-NCM: B2’s advantage exists only at the length it was trained on."));
add(H2("D.6  Plan-only adapters (frozen generator)"));
add(P("Goal: keep the released model’s fluency (F1) by freezing the generator and training only LoRA adapters on the global passes, plus a learned bias on attention to the plan."));
add(bullets([
  "*Crash:* a float attention mask with gradients fails in the fused GPU attention kernel. *Fix:* implement the plan bias as an extra key dimension.",
  "*Bias never opened:* Adam moves a parameter by about one learning-rate step per update, so a bias initialised at −8 with lr 10^−4^ would need 10^5^ steps. *Fix:* initialise at −6, with its own lr 10^−2^.",
  "*Leak:* adapters also changed the prefix and the chunk embeddings of local levels, so the model was no longer “plan-only”. *Fix:* adapters skip the prefix, and chunk embeddings apply to global levels only. Seven unit checks verify this.",
]));
add(P("*Outcome:* **failed**. The adapters have the worst NLL of all fine-tuned models (1.834 vs 1.744 for Ours; chunk 4: 2.22 vs 1.79), because the frozen local levels never learn chunks 2–4. They also generate worst (genPPL 28.5 vs 22.0 at M=4), so freezing did not preserve the released model’s fluency. Full fine-tuning stays the method."));
add(H2("D.7  Layer-drop refiner (negative result, moved to future work)"));
add(P("Goal: cut serving cost by running local levels with 6 of the 12 layers, with each kept layer reading the plan cache of the matching global layer."));
add(bullets([
  "*Untrained:* the best 6-layer pattern ([0, 1, 2, 9, 10, 11]) loses 4.0 nats on levels 4–11.",
  "*Fix attempted:* distillation from the full-depth model (KL weight 1.0, 1,500 steps). This recovered most of the loss, but not enough: +0.50 nats with genPPL 21.6→41.6 (levels 4–11, 1.55× faster per sample); +0.30 nats with genPPL 32.0 (levels 7–11, 1.38× faster).",
]));
add(P("**Not fixed.** Depth reduction is left to follow-up work. Block-NCM’s serving-cost claims are therefore restricted to low batch sizes (Appendix B)."));
add(H2("D.8  Summary"));
add(table("Failures and fixes, by model.", ["model / part", "failure", "fix", "status"],
  [["all (recipe)", "clean fine-tuning: genPPL 47.7 on single chunks", "corruption 0.5, lr 3×10^−5^; residual: unfiltered data + KL anchor", "16.1 (released 16.5), mostly fixed; anchor hurts at M=4"],
   ["all (metrics)", "padding diluted NLL", "mask padding per document", "fixed"],
   ["all (memory)", "dense graph > 8 GB", "per-level path (< 10^−6^ equal)", "fixed"],
   ["all (fp16)", "quantizer NaN, silent", "fp32 quantizer, abort on NaN", "fixed"],
   ["Ours G=3", "seams 1.39", "halo attention, h=4", "1.19, fixed"],
   ["Ours G=3", "plan recomputed per level", "KV-cached plan (exact)", "2.0×, fixed"],
   ["Ours G=5, 6", "degrade at M=6 (23.8, 30.7)", "halo; G=5", "G=5: 20.6 (3 seeds), fixed; G=6: 23.6"],
   ["B2", "collapse at M=6 (33.5, 31.0, 29.4)", "none (baseline)", "finding"],
   ["plan-only adapters", "mask crash; bias stuck; leak; then NLL 1.834, genPPL 28.5", "key-dim bias; own lr; skip prefix", "!failed"],
   ["layer-drop refiner", "+0.3–0.5 nats, genPPL ×1.5–2", "distillation (insufficient)", "!failed"]],
  [1700, 2600, 2600, 2400]));

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "Rishab Gupta",
  title: "Block-NCM: Long Stories in a Constant Number of Parallel Passes",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true }, paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 500, hanging: 280 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 500, hanging: 320 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "blockncm.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
