"""Where do generated stories go wrong, and which mistakes are most common?

Rule-based detectors run over existing samples (no GPU, no new generation). Each error is located by chunk
(1-4) and by position inside its 128-token chunk, so we can see whether errors cluster at chunk seams.
Real long stories (validation split, cut into the same 128-token chunks) give the base rate of each detector,
because some "errors" (e.g. "very very") also occur in real text.

Output: ~/ncm/runs/error_types.json and a printed table.
"""
import glob, json, os, random, re, sys
from collections import Counter, defaultdict
import numpy as np
from omegaconf import OmegaConf
sys.path.insert(0, os.path.expanduser('~/ncm'))
from tokenization.bpe_tokenizer import BPETokenizer

CK = os.path.expanduser('~/ncm/checkpoints/tinystories')
cfg = OmegaConf.load(f'{CK}/config.yaml').tokenizer.pre_tokenizer
cfg.vocab_file_path = f'{CK}/bpe_vocab_8192.pickle'; cfg.merges_file_path = f'{CK}/bpe_merges_8192.pickle'
tok = BPETokenizer.from_config(cfg)
EOS, PAD = tok.eos_token_id, tok.pad_token_id
R = os.path.expanduser('~/ncm/runs')
N, C = 512, 128


def chunks_to_story(chunks):
    """Per-chunk texts (EOS-truncated) and whether the story ended on its own."""
    out, ended = [], False
    for c in chunks:
        ids = [i for i in c if i != PAD]
        if EOS in ids:
            ids, ended = ids[:ids.index(EOS)], True
        out.append(tok.decode(ids, skip_special_tokens=True))
        if ended:
            break
    return out, ended


def from_samples(f):
    return [chunks_to_story(ch) for _, ch in json.load(open(f))['samples']][:N]


def real_val():
    b = glob.glob(os.path.expanduser('~/ncm/**/train_bpe_8192.bin'), recursive=True)[0]
    d = os.path.dirname(b)
    off = np.load(f'{d}/train_bpe_8192_offsets.npy').astype(np.int64)
    t = np.memmap(b, dtype=np.uint16, mode='r')
    starts = off[:-1] if off[0] == 0 else np.concatenate([[0], off[:-1]])
    lens = off[1:] - off[:-1] if off[0] == 0 else np.diff(np.concatenate([[0], off]))
    elig = np.where(lens > 384)[0]
    val = elig[(elig % 100) == 0]
    out = []
    for i in val:
        ids = [int(x) for x in t[starts[i]:starts[i] + 4 * C]]
        out.append(chunks_to_story([ids[k:k + C] for k in range(0, len(ids), C)]))
    random.Random(0).shuffle(out)
    return out[:N]


FEM = set('Lily Anna Mia Sara Sarah Sue Amy Lucy Emma Kitty Jane Mary Lila Ella Zoe Molly Daisy Rose Kim Jenny Susie'.split())
MASC = set('Tom Ben Sam Tim Timmy Max Jack Bob Joe Billy Johnny Leo Jake Mark Spot Tommy Pete Peter Bobby Jim'.split())
NOT_NAME = set("""I Mom Dad Mommy Daddy Grandma Grandpa The A An He She They It We You His Her Their Its Our My Your
Once One Then When But And So If Oh Wow Yes No Okay Ok What Why How Where Who Look Let Please Thank Thanks Hello Hi
Sorry Come Go This That There Here Do Don't Can Can't Will Won't Is Are Was Were Mr Mrs Miss Sir Ow Yay Hooray Good
Bye Stop Help Hey Ouch Yuck Uh Hmm Ha Mama Papa Granny Nana Sure Maybe Today Every Some After Before Now Not Be
Just Very Lets Let's Be Don't Did Does Have Had Has All Look See Wait Hurry Remember Come Thank Mister Teacher""".split())
SAY = r'(?:says|said|asks|asked|cries|cried|yells|yelled|shouts|shouted|replies|replied|answers|answered|whispers|whispered|exclaims|exclaimed)'


def detect(chs, ended):
    """Return list of (type, char_offset) over the joined story, plus seam offsets."""
    text = ''.join(chs)
    seams, pos = [], 0
    for c in chs[:-1]:
        pos += len(c); seams.append(pos)
    errs = []
    # 1. doubled word: "the the", "rainbow rainbow"
    for m in re.finditer(r"\b([A-Za-z']+)\s+\1\b", text, flags=re.I):
        errs.append(('doubled word ("the the")', m.start(2) if m.lastindex and m.lastindex > 1 else m.end() - len(m.group(1))))
    # 2. fused double inside one word: "OwOw", "stopStop", "Don't't"
    for m in re.finditer(r"\b([A-Za-z]{2,})\1\b|n't't\b", text, flags=re.I):
        errs.append(('fused double ("stopStop")', m.start()))
    # 3. paragraph with an odd number of quote marks
    off = 0
    lines = text.split('\n')
    for li, line in enumerate(lines):
        if line.count('"') % 2 == 1:
            errs.append(('unbalanced quotes', off))
        # 4. paragraph that stops without end punctuation (not the last paragraph)
        s = line.strip()
        if s and li < len(lines) - 1 and not re.search(r'[.!?]["\']?$', s):
            errs.append(('paragraph ends mid-sentence', off + len(line)))
        # 5. speaker talks to themself: '"..., Ben," Ben says' / '"No, Anna ..." Anna says'
        for q in re.finditer(r'"([^"\n]+)"\s*([A-Z][a-z]+)\s+' + SAY, line):
            name = q.group(2)
            if name not in NOT_NAME and re.search(r'\b' + name + r'\b', q.group(1)):
                errs.append(('speaker addresses themself', off + q.start(2)))
        off += len(line) + 1
    # 6. pronoun of the wrong gender right after a sentence about one named character
    sents = list(re.finditer(r'[^.!?\n]+[.!?]', text))
    for a, b in zip(sents, sents[1:]):
        sa, sb = a.group().strip().strip('"'), b.group().strip().strip('"')
        ma = re.match(r'([A-Z][a-z]+)\s+(?!and\b)[a-z]+', sa)
        mb = re.match(r'(He|She)\b', sb)
        if ma and mb and ' and ' not in sa[:30]:
            n = ma.group(1)
            if (n in FEM and mb.group(1) == 'He') or (n in MASC and mb.group(1) == 'She'):
                errs.append(('wrong pronoun (Lily ... He)', b.start()))
    # 7. same sentence twice in a row
    for a, b in zip(sents, sents[1:]):
        x, y = a.group().strip().lower(), b.group().strip().lower()
        if x == y and len(x.split()) >= 3:
            errs.append(('sentence repeated', b.start()))
    # 8. a name used only once, after the first chunk (character appears from nowhere)
    names = Counter()
    first = {}
    for m in re.finditer(r'(?<![.!?"\n]\s)(?<!^)\b([A-Z][a-z]+)\b', text):
        pass
    for m in re.finditer(r'\b([A-Z][a-z]+)\b', text):
        w = m.group(1)
        if w in FEM or w in MASC:
            names[w] += 1; first.setdefault(w, m.start())
    for w, cnt in names.items():
        if cnt == 1 and chs and first[w] >= len(chs[0]):
            errs.append(('one-off name after chunk 1', first[w]))
    # 9. story ends on its own but mid-sentence
    if ended and text.strip() and not re.search(r'[.!?]["\']?$', text.strip()):
        errs.append(('story ends mid-sentence', len(text.rstrip())))
    return errs, seams, text


def locate(o, chs):
    """chunk index (0-based) and relative position (0..1) inside that chunk for char offset o."""
    pos = 0
    for k, c in enumerate(chs):
        if o < pos + len(c) or k == len(chs) - 1:
            return k, min(max((o - pos) / max(len(c), 1), 0), 1)
        pos += len(c)


def analyse(stories):
    n_words = 0
    per_type = Counter(); stories_with = Counter()
    chunk_of = defaultdict(Counter); quart = defaultdict(Counter); near_seam = Counter()
    seam_chars = tot_chars = 0
    for chs, ended in stories:
        errs, seams, text = detect(chs, ended)
        n_words += len(re.findall(r"[A-Za-z']+", text))
        tot_chars += len(text)
        seam_chars += sum(min(s + 40, len(text)) - max(s - 40, 0) for s in seams)
        seen = set()
        for t, o in errs:
            per_type[t] += 1; seen.add(t)
            k, r = locate(o, chs)
            chunk_of[t][k] += 1; quart[t][min(int(r * 4), 3)] += 1
            if any(abs(o - s) <= 40 for s in seams):
                near_seam[t] += 1
        for t in seen:
            stories_with[t] += 1
    return dict(n_words=n_words, n=len(stories), per_type=per_type, stories_with=stories_with,
                chunk_of=chunk_of, quart=quart, near_seam=near_seam, seam_share=seam_chars / max(tot_chars, 1))


SRC = {'real': None,
       'B0': f'{R}/b0c_M4_G0/eval_M4_G0_t1.0_samples.json',
       'G3+h4': f'{R}/oursc_M4_G3_halo4/eval_M4_G3_t1.0_samples.json',
       'G5+h4': f'{R}/oursc_M4_G5_halo4/eval_M4_G5_t1.0_samples.json',
       'B2': f'{R}/b2c_M4_G11/eval_M4_G11_t1.0_samples.json',
       'G5+h4 M=6': f'{R}/oursc_M4_G5_halo4/eval_M6_G5_t1.0_samples.json',
       'B2 M=6': f'{R}/b2c_M4_G11/eval_M6_G11_t1.0_samples.json'}

if __name__ == '__main__':
    res, data = {}, {}
    for name, f in SRC.items():
        data[name] = real_val() if f is None else from_samples(f)
        res[name] = analyse(data[name])
    types = sorted({t for r in res.values() for t in r['per_type']}, key=lambda t: -res['G5+h4']['per_type'][t])
    names = list(SRC)
    print('errors per 1,000 words (share of stories with at least one)')
    print(f"{'type':34s}" + ''.join(f'{n:>16s}' for n in names))
    out = {'per_1k_words': {}, 'share_of_stories': {}, 'by_chunk_G5': {}, 'by_quarter_G5': {}, 'near_seam_G5': {},
           'seam_char_share_G5': res['G5+h4']['seam_share']}
    for t in types:
        row = f'{t:34s}'
        out['per_1k_words'][t] = {}; out['share_of_stories'][t] = {}
        for n in names:
            r = res[n]
            v = 1000 * r['per_type'][t] / r['n_words']; s = r['stories_with'][t] / r['n']
            out['per_1k_words'][t][n] = round(v, 2); out['share_of_stories'][t][n] = round(s, 3)
            row += f'{v:9.2f} ({100*s:3.0f}%)'
        print(row)
    print(f"\nG5+h4: where the errors are (chunk 1-4; quarter of the chunk; share within 40 chars of a seam,"
          f" text share near seams = {100*res['G5+h4']['seam_share']:.0f}%)")
    g = res['G5+h4']
    for t in types:
        tot = g['per_type'][t]
        if not tot:
            continue
        ch = [g['chunk_of'][t][k] for k in range(4)]; q = [g['quart'][t][k] for k in range(4)]
        out['by_chunk_G5'][t] = ch; out['by_quarter_G5'][t] = q; out['near_seam_G5'][t] = round(g['near_seam'][t] / tot, 3)
        print(f'{t:34s} n={tot:4d}  chunks {ch}  quarters {q}  near seam {100*g["near_seam"][t]/tot:.0f}%')
    out['n_words'] = {n: res[n]['n_words'] for n in names}
    json.dump(out, open(f'{R}/error_types.json', 'w'), indent=1)
    # stories for reading by hand, chunk seams marked with ‖
    rnd = random.Random(1)
    pick = rnd.sample(range(len(data['G5+h4'])), 20)
    with open(f'{R}/error_read_G5.txt', 'w') as fh:
        for i in pick:
            chs, ended = data['G5+h4'][i]
            fh.write(f'=== story {i} ({"ended" if ended else "cut at 512"})\n' + ' ‖ '.join(chs).strip() + '\n\n')
