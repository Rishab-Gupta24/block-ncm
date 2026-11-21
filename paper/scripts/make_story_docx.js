// Story-style engineering log of the Block-NCM project (same facts as RESULTS.md and the paper, told as a journey).
//   node paper/scripts/make_story_docx.js      -> paper/blockncm_story.docx   (needs npm "docx"; NODE_PATH if elsewhere)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel, WidthType,
  BorderStyle, ShadingType, LevelFormat, PageNumber, Footer, TableLayoutType,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const FONT = "Calibri";
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1300;

function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), superScript: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));
const P = (t) => new Paragraph({ children: runs(t), spacing: { after: 140, line: 288 } });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullets = (items) => items.map((t) => new Paragraph({ children: runs(t), numbering: { reference: "bullets", level: 0 },
  spacing: { after: 70, line: 276 } }));
let inst = 0;
const numbered = (items) => { const i = ++inst; return items.map((t) => new Paragraph({ children: runs(t),
  numbering: { reference: "numbers", level: 0, instance: i }, spacing: { after: 70, line: 276 } })); };
// a shaded "side note" box
const note = (t) => new Paragraph({ children: runs(t, { size: 21 }), spacing: { before: 80, after: 180, line: 276 },
  shading: { fill: "FFF4E0", type: ShadingType.CLEAR, color: "auto" }, indent: { left: 200, right: 200 },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: "F0A030", space: 8 } } });
let figN = 0;
function figure(file, widthIn, caption) {
  const b = fs.readFileSync(path.join(ROOT, "figures", file));
  const w = b.readUInt32BE(16), h = b.readUInt32BE(20);
  const wpx = Math.round(widthIn * 96), hpx = Math.round((wpx * h) / w);
  figN += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: b, transformation: { width: wpx, height: hpx },
        altText: { title: `Figure ${figN}`, description: caption.slice(0, 200), name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 },
      children: [new TextRun({ text: `Figure ${figN}: `, bold: true, size: 19, color: "555555" }),
        ...runs(caption, { size: 19, italics: true, color: "555555" })] }),
  ];
}
const bd = { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" };
function table(header, rows, widths) {
  const mk = (r, hdr) => new TableRow({ tableHeader: hdr, cantSplit: true, children: r.map((t, i) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders: { top: bd, bottom: bd, left: bd, right: bd },
    shading: hdr ? { fill: "2F5D8A", type: ShadingType.CLEAR, color: "auto" } : (i === 0 ? { fill: "F3F6FA", type: ShadingType.CLEAR, color: "auto" } : undefined),
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ keepNext: true, alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(t.replace(/^!/, ""), { size: 19, bold: hdr || t.startsWith("!"), color: hdr ? "FFFFFF" : undefined }) })],
  })) });
  return [new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
    layout: TableLayoutType.FIXED, alignment: AlignmentType.CENTER, rows: [mk(header, true), ...rows.map((r) => mk(r, false))] }),
    new Paragraph({ spacing: { after: 160 }, children: [] })];
}

// ====================================================================== the story
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({
  text: "Technical Engineering Log: Teaching a 128-Token Model to Tell a Whole Story", bold: true, size: 36, color: "1F3864" })] }));
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({
  text: "Block-NCM, or: how I made a fast model write long stories without making it slow", italics: true, size: 24 })] }));
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 }, children: [new TextRun({
  text: "Rishab Gupta  ·  19–20 November 2026", size: 22, color: "666666" })] }));

add(H1("Where did my idea start?"));
add(P("Most AI models that write text are like a very careful person typing with one finger. They write one word, look at everything so far, write the next word, look again, and so on. A 500-token story means about 500 trips through the model. These are called *autoregressive* (AR) models, and almost every chatbot you know works this way."));
add(P("Then in October 2026 HazyResearch (Stanford) released something called **Next-Concept Models**, or NCM. NCM does not type word by word. It first sketches the *rough idea* of a chunk of text, then adds more detail, then more, like an artist who draws a rough outline, then the shapes, then the shading. Each level of detail is drawn all at once, in one pass. A 128-token chunk takes just **11 passes**, no matter what. That is why NCM is so fast."));
add(P("But there was a catch, and that catch became this whole project. NCM only ever sees **128 tokens at a time**. That is about one paragraph of a children’s story. So I asked myself the obvious question: *what happens when you want a whole story, not one paragraph?*"));
add(note("**Side note: what is a “token”?** A token is a piece of a word. “playground” might be two tokens, “play” and “ground”. 128 tokens is roughly 90–100 words."));

add(H1("Meet the base NCM: what Stanford built, what they claim, and where it gets stuck"));
add(P("Before I break something, I should explain how it works. So here is the original NCM from Stanford’s HazyResearch lab (Kumbong et al., arXiv 2610.14162), the model this whole project stands on."));

add(H2("Two parts: a tokenizer that compresses, and a generator that invents"));
add(P("NCM is really two models glued together:"));
add(bullets([
  "**The tokenizer (a VQ-VAE)** squeezes one 128-token paragraph into a stack of codes and can turn the codes back into text. Think of it as a zip file for paragraphs.",
  "**The generator (a transformer, about 85M parameters)** invents new code stacks from scratch. The tokenizer then turns them into a story.",
]));
add(figure("story_ncm_pipeline.png", 6.2, "The base NCM. During training, real paragraphs are compressed into codes. When writing, the generator invents the codes level by level and the decoder turns them into text."));
add(P("The clever bit is *how* the codes are stacked. After a single starting code come 11 levels, each longer than the last: 8, 16, 24, 32, 40, 52, 64, 76, 92, 108 and 128 codes. Counting the start, that is **641 codes** for one paragraph. The first levels hold the gist (who is in the story, what the mood is); the last levels pin down the exact words. This trick is called **residual quantisation**: each level only stores what the levels above it got wrong."));
add(note("**Side note: residual quantisation, the shopkeeper way.** Say you owe ₹487. You first hand over the biggest note that fits (₹200, ₹200), then a smaller one for what is left (₹50, ₹20, ₹10), then coins (₹5, ₹2). Every step only pays the *leftover*. NCM does the same with meaning: level 1 pays the big part, and each later level pays a smaller leftover, until level 11 makes it exact."));

add(H2("Why it is fast: a whole level in one go"));
add(P("An AR model writes token 1, then token 2, then token 3: 128 trips through the model for 128 tokens. NCM’s generator instead predicts **every code of a level at the same time**, in one pass. Level 1 in one pass, level 2 in the next, and level 11 last: 11 passes write the whole paragraph, every time."));
add(figure("story_ncm_passes.png", 6.2, "Left: an AR model needs one pass for every token, 128 passes for a paragraph. Right: NCM needs one pass per level, 11 passes, and the work in each pass is done in parallel."));

add(H2("What the HazyResearch team claims"));
add(P("Here is what their paper says, and what I could check myself:"));
add(bullets([
  "**Fixed, tiny number of passes:** 128 tokens in 11 passes, whatever the text. *(Checked: true by design, it is right there in the code.)*",
  "**Much faster than diffusion language models:** 10–31× faster at comparable quality, on three benchmarks. Diffusion models also write in parallel, but need many more refinement steps. *(Not re-checked by me; I only compared against AR models.)*",
  "**Quality on TinyStories:** genPPL **15.41**. *(Checked: I got 16.38 ± 0.22 over five runs, close enough.)*",
  "**It scales up:** they also release a bigger model for web text (OpenWebText) that writes 1,024 tokens in 32 levels (1, 4, 9, … 1,024 codes).",
  "**A key training trick:** conditioning corruption (random junk mixed into the inputs while training), which keeps the model calm when it makes its own mistakes. *(Checked the hard way, as you will see on Day 1.)*",
]));
add(P("And to their credit, they are honest about the price: their own AR baseline scores **8.89** on the same test, better than NCM’s 15.41."));

add(H2("The bottlenecks (a.k.a. why this project exists)"));
add(figure("story_ncm_tradeoffs.png", 6.2, "The price of speed. Left: NCM writes about 1.7× worse text than an AR model of similar size (genPPL, lower is better; the first three bars are from the NCM paper). Right: NCM computes about 5 positions for every token it writes."));
add(numbered([
  "**The 128-token wall.** The TinyStories tokenizer and generator only ever see one paragraph. Want a whole story? Nothing in the release does that. **This is the one I attack in Paper 1.**",
  "**The quality gap.** About 1.7× worse than AR by genPPL (15.41 vs 8.89). Every model I build on top of NCM inherits this gap. **This is Paper 2’s job.**",
  "**More work per token.** 641 codes for 128 tokens means about 5 positions computed per token, against 1 for AR. With one story at a time this hardly matters: the number of passes decides the speed. But when a GPU writes many stories at once, AR catches up, because then the total work decides the speed.",
  "**The same 11 levels for everyone.** An easy paragraph gets as many passes as a hard one. The NCM paper itself says an input-dependent schedule “would adapt compute to example complexity”. My quick probe agrees: keeping only levels 6, 9 and 11 still rebuilds 99.5% of tokens, with 39% fewer positions. But the last level can never be skipped.",
  "**Release gaps.** The code ships with no training script and no loss function (I wrote both), one bug to patch first (issue #11), and an original training recipe (batch 2,048, 50,000 updates) that will never fit on an 8 GB laptop GPU.",
]));
add(P("Bottleneck 1 looked like the cleanest place to start: if NCM can only write one paragraph, can I teach it to write many paragraphs *without giving up its 11 passes*? That question is the rest of this log."));

add(H1("What I wanted, and is it even possible on my laptop?"));
add(P("The goal: make NCM write stories four times longer (512 tokens) and keep its superpower, the small fixed number of passes. If a longer story needs more passes, the whole point of NCM is gone."));
add(P("Now the bottleneck, which every student researcher knows too well: **hardware and budget**. I do not have a lab full of GPUs. I have my laptop:"));
add(bullets([
  "**ROG Strix G16** with an AMD Ryzen 9 8940HX",
  "**16 GB RAM**",
  "**NVIDIA GeForce RTX 5050 Laptop GPU with 8 GB** of memory",
]));
add(P("Is it possible? Yes, with some proper *jugaad*. When my laptop could not do something, I used the **free tier of Kaggle**, which gives you two old NVIDIA T4 GPUs for free. Total money spent on this whole project: **zero rupees**. Total sleep lost: a lot more than zero."));

// ---------------------------------------------------------------- Day 1
add(H1("DAY 1 – The Start (19 November)"));
add(H2("Step zero: does the original model even work on my machine?"));
add(P("Before changing anything, I had to make sure I could run the original NCM exactly like the authors did. I measured its quality with a score called **genPPL**. A big judge model (GPT-2 Large) reads the generated text and says how surprised it is. **Lower is better**: less surprise means more natural text."));
add(P("My result: **16.38**, across five runs. The paper says 15.41, so I was in the same neighbourhood. Good, the engine starts. Now let’s modify it."));

add(H2("Three ways to write a longer story"));
add(P("I imagined a story split into 4 chunks, and four writers, one per chunk. There are three ways to organise them:"));
add(numbered([
  "**B1, one after another:** writer 2 waits for writer 1 to finish, writer 3 waits for writer 2, and so on. The story connects well, but 4 chunks now take 4 × 11 = 44 passes. Slow. This kills the whole point of NCM, so I only use it as a speed reference.",
  "**B0, separate rooms:** all four writers work at the same time but never talk. Super fast (11 passes), but chunk 2 has no idea what happened in chunk 1. Characters change names, the plot teleports.",
  "**B2, one giant group chat:** everyone sees everything at every level of detail. Still 11 passes. Sounds perfect, right? Hold that thought, there is a plot twist coming.",
]));
add(P("And my idea, **Block-NCM**, sits in between. Think of a writers’ room for a TV show:"));
add(bullets([
  "First, everyone sits together and agrees on the **rough plot**: who the characters are and what happens. In NCM terms, the first few coarse levels (I call the number of these levels **G**) are shared across all chunks. This is the **plan**.",
  "Then each writer goes off and writes **their own chunk** in detail. These are the *local* levels.",
  "And each writer is allowed to **peek at the last few lines of the previous chunk and the first few lines of the next one**, so the sentences join up smoothly. I call this peek the **halo**. It costs almost nothing.",
]));
add(figure("fig_architecture.png", 6.2, "Who talks to whom. Each row is one level of detail (one pass). Blue bars: levels shared across all chunks. Grey boxes: each chunk on its own. Orange edges: the halo peek. All three designs take 11 passes for any number of chunks."));

add(H2("First disaster: I broke the model"));
add(P("I fine-tuned the model on long TinyStories (stories longer than 384 tokens, about 125,000 of them), ran it, and checked the score. The 4-chunk stories scored around **50**. And not just the long stories: even single paragraphs scored **47.7**, against 16.5 for the original. I had made it about three times worse. The text looked like *“there was a that threat”*. Fluent-sounding, but nonsense."));
add(P("The weird part: by the training numbers, my model looked *better* than the original. It predicted the training text better, but wrote worse. It was like a student who memorised the textbook word for word but cannot write an essay."));
add(P("After a lot of digging I found the reason in the NCM paper’s appendix. The original model was trained with **conditioning corruption**: during training, some of the inputs are deliberately replaced with random junk. It is like practising cricket with someone throwing random balls at you. You get used to mistakes, so when the model makes its own small mistakes while writing, it does not panic. I had trained on perfectly clean inputs, so my model panicked at its first mistake and fell apart."));
add(P("I added the corruption back, tested a few learning rates on small quick runs, and the score went from **47.7 to 20.7**. Much better, but still not 16.5. That 4-point mystery haunted me for a while (spoiler: the answer comes later and it is hilarious)."));

add(H2("The first real results, and the seams problem"));
add(P("With the fixed recipe, my first Block-NCM (G=3, plan made of the 3 coarsest levels) had the **best training score** of all three designs. But when it actually wrote stories, the giant group chat (B2) was better: 19.0 against my 22.0."));
add(P("Why? The **seams**. I measured how natural the text is right at the boundary between two chunks, compared with the middle of a chunk. Real stories score 1.01 on this *seam ratio* (no visible seam). B0 scored 1.52, B2 1.17, and mine 1.39. My plan knew the characters and the topic, but a sentence that starts at the end of chunk 1 still had no way to finish in chunk 2."));
add(P("So I added the **halo**: let each chunk peek at 4 positions on each side of its neighbours. Result: the seam ratio dropped from **1.39 to 1.19**, closing 95% of the gap to B2. And it also gave the best training score of every run. A tiny peek fixed most of the problem."));

add(H2("The failure montage"));
add(P("Not everything worked. Here is my wall of shame, all honestly reported in the paper:"));
add(bullets([
  "**The layer-drop refiner:** run the detailed levels with only half of the model’s layers to go faster. It was 1.55× faster, and the score went from 21.6 to 41.6. Fast and terrible.",
  "**Cut chunks at sentence ends:** sounds smart, right? Seams should be easier. Both models got worse (24.1 → 27.0).",
  "**Freeze the model, train only small adapters:** to keep the original model’s fluency. It ended up the worst model of all (28.5).",
  "**A “KL anchor” to the original model:** helped single paragraphs (21.0 → 18.0) but made the 4-chunk models worse (19.1 → 25.7). The original model only knows single paragraphs, so it kept pulling my model away from using the plan.",
]));

add(H2("Night 1: moving to Kaggle, and the plot twist"));
add(P("My laptop could only train one model at a time, about an hour each. So I moved runs to Kaggle’s free T4s. First problem: the T4 does not support the number format my laptop used, and in the other format part of the model produced NaN (“not a number”) and **kept training silently on garbage**. Fix: run that part in full precision, and stop training the moment anything becomes NaN."));
add(P("Then came the plot twist. I tested all models on stories **longer than they were trained on**: 6 chunks (768 tokens) instead of 4. I expected everyone to get a bit worse. Instead:"));
add(bullets([
  "The giant group chat, **B2, collapsed: 19.0 → 33.5**.",
  "My Block-NCM with G=3 and halo **barely moved: 21.0 → 21.5**.",
]));
add(P("The design that won at the trained length fell apart one step beyond it. That was the moment this project went from “a slightly worse B2” to “actually interesting”."));

// ---------------------------------------------------------------- Day 2
add(H1("DAY 2 – Proving it is real (20 November)"));
add(H2("One result is luck, three results is science"));
add(P("I queued two more random seeds on Kaggle and used Claude Code as my away-from-keyboard runner—starting the next job whenever one finished while I slept or was at a hackathon. When I came back:"));
add(table(["", "B0 (separate rooms)", "Ours, G=3 + halo", "Ours, G=5 + halo", "B2 (group chat)"],
  [["Score, 4 chunks (trained)", "24.2", "21.1", "19.2", "!18.7"],
   ["Score, 6 chunks (never trained)", "23.8", "21.2", "!20.6", "31.3"],
   ["Seam ratio, 4 chunks", "1.49", "1.20", "1.14", "!1.12"]],
  [2500, 1600, 1600, 1600, 1600]));
add(P("*(genPPL, lower is better, average of three training runs.)* B2’s collapse showed up in **all three** runs: 33.5, 31.0, 29.4. And B2 still wins at the trained length, but only by 0.5 points over my G=5 model."));
add(figure("fig_tradeoff.png", 6.2, "(a) At the trained length B2 (red) is best. One step beyond it, B2 shoots up while Block-NCM (blue, green) stays flat. (b) How much worse the very first chunk gets when the model is asked for 6 chunks instead of 4."));

add(H2("The “wait, is this too good to be true?” moment"));
add(P("Then I found something that made my heart sink. Most generated stories **end around chunk 4** anyway, because the training stories were cut at 512 tokens. Models that stopped on time were scored on less text, and models that kept rambling were scored on more. Maybe B2 was not “collapsing”, maybe it just did not know when to stop?"));
add(P("So I designed a fair test. Compare the **same chunk** in both cases: chunk 1 of a 4-chunk story against chunk 1 of a 6-chunk story. Chunk 1 is the opening paragraph. Nothing has been written before it, and no story can “run on” inside it. If chunk 1 gets worse, the problem is the plan itself."));
add(P("Result: B2’s opening paragraph got **71% worse** (20.6 → 35.3) just because it was *told* the story would be 6 chunks long. Mine got 6% worse. B0, which has no shared plan, did not change at all. The more levels shared across chunks, the bigger the damage: 0%, 6%, 35%, 71%."));
add(note("**Analogy:** imagine a director who planned a 4-episode series being told it is now 6 episodes. B2 is the director who panics so much that even episode 1 comes out worse. Block-NCM only shares the rough outline, so its writers barely notice."));
add(figure("fig_chunks.png", 3.6, "Each chunk’s score in 4-chunk stories (solid) and 6-chunk stories (dashed). B2 (red) is worse in every chunk, including the first."));

add(H2("Showing it with actual text"));
add(P("Numbers are nice, but here is what it looks like. These are real outputs, errors included, from the same random seed, at the point where chunk 1 ends (‖) and chunk 2 begins:"));
add(note("**B0 (separate rooms):** … “Hey, my my car is faster than yours!” Sam says, pointing to his bike. ‖ Okay, go,”,” Ben said. … They climbed stairs to the other side of the slide. …  → *Sam disappears, a new story starts.*"));
add(note("**Block-NCM:** … “Ow, ow, Billy!” Lila cries, rubbing her arm ‖ . Ben runs to the puddle and hugs her. He lets Sara go away. …  → *the sentence continues across the seam, but a new name (Sara) sneaks in.*"));
add(note("**B2 (group chat):** … “Ow, ow, ow! That hurts!” Lila cried, poking her finger. ‖ “Ow, that hurts!” Ben said. …  → *keeps both characters.*"));

add(H2("The speed race"));
add(P("Now the part that matters for Paper 1: **speed**. I raced my Block-NCM against TinyStories-33M, a normal AR model trained on the same kind of stories, on the same laptop GPU, both writing a 512-token story."));
add(P("First run: at batch size 32 my model came out *slower* than the AR model. What?! After profiling, the culprit was tiny and sneaky. The last step that turns codes back into words ran in full 32-bit precision, which hits a slow path on this GPU: **4.2 seconds** for something that should take 0.01. The jugaad fix was half precision for that one step, which still matches the full-precision output on 99.96% of words."));
add(figure("fig_speed.png", 3.6, "Top: one story at a time, as the story gets longer. Bottom: 512-token stories, as I write more of them at once. My RTX 5050 laptop GPU."));
add(bullets([
  "**One story at a time: 0.38 s vs 6.20 s, so Block-NCM is 16.5× faster.**",
  "**8 stories at once: 10.8×. 16 at once: 6.0×. 32 at once: 3.2×.** The AR model catches up when it can share its work across many stories.",
]));
add(P("And my model is actually *bigger* than the AR model (546 MB vs 274 MB). It wins because it makes 11 passes instead of about 517, not because it is smaller."));

add(H2("Day 3 bonus: the race against B1, and the memory leak that wasn’t"));
add(P("One claim in the paper was still an estimate: that writing the chunks one after another (B1) is slower. So in the early hours of 21 November the laptop raced them while I slept. B1 has no trained model, but speed doesn’t care about the weights, so I just ran the single-paragraph model 2, 4 and 8 times in a row."));
add(bullets([
  "**2 chunks:** Block-NCM 0.39 s, B1 0.80 s (2.1× slower)",
  "**4 chunks:** Block-NCM 0.36 s, B1 1.56 s (4.3× slower)",
  "**8 chunks:** Block-NCM 0.38 s, B1 3.56 s (**9.4× slower**)",
]));
add(P("Look at Block-NCM’s column: it doesn’t move. Two paragraphs or eight, same 11 passes, same time. That flat line is the whole paper in one picture. (And B1’s real times would be even worse, because a real B1 also has to read the earlier chunks.)"));
add(note("**Plot twist #2:** at 32 stories × 8 chunks, Block-NCM took **25 seconds**. Huh? It turned out PyTorch was hoarding GPU memory in fragments: for 32 stories × 4 chunks it *reserved* 8.6 GB for a job that only *used* 3.6 GB. My laptop GPU has 8 GB, so Windows quietly spilled the rest into normal RAM, which is painfully slow. One setting (expandable_segments) fixed it: the same job now fits in 3.8 GB, and the batch-32 speedup over the AR model went from 2.3× to **3.2×**. My old number had been unfair to my own model! (32 stories × 8 chunks still doesn’t fit in 8 GB. Some limits are real.)"));

add(H2("Solving the 4-point mystery"));
add(P("Remember the 20.5 vs 16.5 gap from Day 1? Here is the hilarious answer. Long TinyStories are written in a *different style*. **0%** of stories longer than 384 tokens start with “Once upon a time”, against 58% of all stories. The judge model simply finds the long-story style a bit more surprising. When I trained on stories of all lengths instead, single-paragraph quality came back to **17.0** (16.1 with the KL anchor), basically the original model. My recipe was mostly fine. My *data* had a different accent."));

add(H2("The honest part"));
add(P("Block-NCM writes text that is about **2× worse** than the AR model by this score (19.2 vs 9.9). But most of that gap was already there in the original NCM (16.4 vs 9.7 on a single paragraph), and the NCM paper’s own table shows the same thing. My job in Paper 1 was to make NCM write *long* stories without losing its speed. Making the text itself better is the next chapter."));

// ---------------------------------------------------------------- wrap-up
add(H1("Final scoreboard"));
add(table(["Question", "Answer"],
  [["Can NCM write 512-token stories in the same 11 passes?", "!Yes: Block-NCM"],
   ["How much faster than a normal AR model?", "!16.5× (one story), 3.2× (32 stories)"],
   ["Does it slow down for longer stories?", "!No: 0.36–0.39 s; B1 is up to 9.4× slower"],
   ["Is it as good as the “group chat” design at the trained length?", "Almost: 19.2 vs 18.7"],
   ["What happens one step beyond the trained length?", "!B2 collapses (31.3), Block-NCM holds (21.2)"],
   ["Is the collapse real or a measuring trick?", "Real: even chunk 1 gets 71% worse for B2"],
   ["Is the text as good as a normal AR model?", "No, about 2× worse (mostly inherited)"],
   ["Money spent", "!₹0 (laptop + free Kaggle)"]],
  [5600, 3700]));

add(H1("What’s next?"));
add(bullets([
  "**Paper 1 (this one): speed and length.** Next stop: my professor’s desk for checking, then publishing.",
  "**Paper 2: quality.** I already have a teaser: letting the model sample 8 rough plans and keep the best one improves the score from **19.4 to 17.1** with no extra training. Writing 8 whole stories and letting a small AR model pick the best gets to **15.7**, but costs 8× the work. Other ideas on the list: training on stories of all lengths, longer training, and learning from a stronger AR model.",
  "**Later:** coding and other domains, once the first papers are checked and published.",
]));
add(P("The biggest lesson? The design that looks best on the test you trained for can be the one that breaks first when the world changes a little. Sometimes the smarter move is to share less, and peek just enough."));

// ====================================================================== document
const doc = new Document({
  creator: "Rishab Gupta", title: "Teaching a 128-Token Model to Tell a Whole Story",
  styles: {
    default: { document: { run: { font: FONT, size: 23 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 320, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 25, bold: true, color: "2F5D8A" }, paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 520, hanging: 300 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 520, hanging: 320 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888" })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = process.env.STORY_OUT || path.join(ROOT, "blockncm_story.docx");  // STORY_OUT: e.g. while the docx is open in Word
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
