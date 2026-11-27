"""Near-duplicate plots: for each story, the most similar other story (TF-IDF cosine over content words,
names removed) within a set of N stories. Reports the share of stories that have a near-twin."""
import glob, json, math, os, re, sys, random
from collections import Counter
import numpy as np
from omegaconf import OmegaConf
sys.path.insert(0, os.path.expanduser('~/ncm'))
from tokenization.bpe_tokenizer import BPETokenizer

CK = os.path.expanduser('~/ncm/checkpoints/tinystories')
cfg = OmegaConf.load(f'{CK}/config.yaml').tokenizer.pre_tokenizer
cfg.vocab_file_path = f'{CK}/bpe_vocab_8192.pickle'; cfg.merges_file_path = f'{CK}/bpe_merges_8192.pickle'
tok = BPETokenizer.from_config(cfg)
EOS, PAD = tok.eos_token_id, tok.pad_token_id
N = 512
R = os.path.expanduser('~/ncm/runs')

def from_samples(f):
    out = []
    for _, chunks in json.load(open(f))['samples']:
        ids = [i for c in chunks for i in c if i != PAD]
        if EOS in ids: ids = ids[:ids.index(EOS)]
        out.append(tok.decode(ids, skip_special_tokens=True))
    return out[:N]

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
        ids = [int(x) for x in t[starts[i]:starts[i] + 512] if int(x) not in (PAD,)]
        if EOS in ids: ids = ids[:ids.index(EOS)]
        out.append(tok.decode(ids, skip_special_tokens=True))
    random.Random(0).shuffle(out)
    return out[:N], len(val)

STOP = set("""a an the and or but so if then than that this these those there here it its it's he she they them
his her hers their we us our you your i me my mine is was were are be been being am do did does done have has had
to of in on at by for with from up down out over under into onto off about as not no yes very too also just
can could will would should may might must shall said say says asked ask one two all some any every each more most
much many little big go went goes going come came get got make made see saw look looked want wanted like liked
know knew think thought day time once upon what when where who why how which while because after before again
back away now then too oh wow let lets let's s t don't didn't can't won't im i'm""".split())

def words(text):
    out = []
    for w in re.findall(r"[A-Za-z']+", text):
        if w[0].isupper() and w.lower() not in STOP:  # names (and sentence-initial content words) dropped
            continue
        w = w.lower().strip("'")
        if len(w) > 2 and w not in STOP:
            out.append(w[:-1] if w.endswith('s') and len(w) > 4 else w)
    return out

sources = {}
real, nval = real_val(); sources['Real long stories (val)'] = real
for name, f in [('AR TinyStories-33M', f'{R}/ar_baseline/ar_M4_t1.0_samples.json'),
                ('B0 (G=0)', f'{R}/b0c_M4_G0/eval_M4_G0_t1.0_samples.json'),
                ('Block-NCM G=3+h4', f'{R}/oursc_M4_G3_halo4/eval_M4_G3_t1.0_samples.json'),
                ('Block-NCM G=5+h4', f'{R}/oursc_M4_G5_halo4/eval_M4_G5_t1.0_samples.json'),
                ('B2 (G=11)', f'{R}/b2c_M4_G11/eval_M4_G11_t1.0_samples.json')]:
    if os.path.exists(f): sources[name] = from_samples(f)
    elif name.startswith('AR'):
        g = f'{R}/ar_baseline/ar_texts.json'
        if os.path.exists(g): sources[name] = json.load(open(g))[:N]

docs = {k: [words(t) for t in v] for k, v in sources.items()}
df = Counter(w for v in docs.values() for d in v for w in set(d))
ND = sum(len(v) for v in docs.values())
vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= 2))}
idf = np.zeros(len(vocab))
for w, i in vocab.items(): idf[i] = math.log(ND / df[w])

def mat(ds):
    X = np.zeros((len(ds), len(vocab)), dtype=np.float32)
    for r, d in enumerate(ds):
        for w, c in Counter(d).items():
            if w in vocab: X[r, vocab[w]] = (1 + math.log(c)) * idf[vocab[w]]
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X

TAUS = [0.3, 0.4, 0.5, 0.6]
res, ex = {}, {}
for k, ds in docs.items():
    X = mat(ds); S = X @ X.T; np.fill_diagonal(S, -1)
    best = S.max(1); arg = S.argmax(1)
    iu = np.triu_indices(len(ds), 1)
    res[k] = {'n': len(ds), 'mean_words': float(np.mean([len(d) for d in ds])),
              'mean_pair_sim': float(S[iu].mean()), 'median_best': float(np.median(best)),
              **{f'twin>={t}': float((best >= t).mean()) for t in TAUS}}
    order = np.argsort(-best)
    picks = {}
    for band in [(0.6, 1.1), (0.5, 0.6), (0.4, 0.5), (0.3, 0.4)]:
        c = [i for i in order if band[0] <= best[i] < band[1]]
        if c: i = c[len(c) // 2]; picks[f'{band[0]}-{band[1]}'] = (float(best[i]), sources[k][i][:700], sources[k][arg[i]][:700])
    ex[k] = picks
print('real val stories available:', nval)
print(f"{'source':28s} {'n':>4s} {'words':>6s} {'meanpair':>8s} {'medbest':>7s} " + ' '.join(f'tw>={t}' for t in TAUS))
for k, r in res.items():
    print(f"{k:28s} {r['n']:4d} {r['mean_words']:6.1f} {r['mean_pair_sim']:8.3f} {r['median_best']:7.3f} " +
          ' '.join(f"{100*r[f'twin>={t}']:7.1f}%" for t in TAUS))
json.dump({'results': res, 'examples': ex}, open(os.path.expanduser('~/ncm/runs/plot_dup.json'), 'w'), indent=1)
