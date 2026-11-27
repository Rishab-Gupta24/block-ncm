"""How much of the genPPL gap is the doubled-word stutter? Re-score existing samples with GPT-2 Large after
removing exact adjacent repeats ("and and" -> "and"), using the same generation_perplexity call as sample_eval.py.

Usage: python experiments/dedup_test.py [--device cpu]
Output: ~/ncm/runs/dedup_test.json
"""
import argparse, json, os, re, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import error_types as E  # tokenizer, sample loading, real text
sys.path.insert(0, os.path.expanduser('~/ncm'))
from evaluation.gen_ppl import generation_perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--device', default='cpu')
p.add_argument('--bs', type=int, default=4)
a = p.parse_args()
torch.set_num_threads(min(16, os.cpu_count()))

DUP = re.compile(r"\b([A-Za-z][A-Za-z']*)([ \t]+)\1\b", flags=re.I)


def dedup(t):
    while True:
        n = DUP.sub(r'\1', t)
        if n == t:
            return t
        t = n


rtok = AutoTokenizer.from_pretrained('gpt2-large')
rlm = AutoModelForCausalLM.from_pretrained('gpt2-large').eval()


def gppl(ts):
    ts = [t for t in ts if t.strip()]
    return float(generation_perplexity(rlm, rtok, [''] * len(ts), ts, batch_size=a.bs, device=a.device).mean_ppl)


def load(f):
    d = json.load(open(f))
    out = []
    for seed, chunks in d['samples']:
        chs, _ = E.chunks_to_story(chunks)
        out.append((seed, ''.join(chs)))
    return out


res = {}
SRC = {'G5+h4 M=4': E.SRC['G5+h4'], 'G5+h4 M=6': E.SRC['G5+h4 M=6'], 'real': None}
for name, f in SRC.items():
    data = [(0, ''.join(chs)) for chs, _ in E.real_val()] if f is None else load(f)
    seeds = sorted({s for s, _ in data})
    row = {'per_seed': []}
    for s in seeds:
        ts = [t for s_, t in data if s_ == s]
        dd = [dedup(t) for t in ts]
        removed = sum(len(t.split()) - len(d.split()) for t, d in zip(ts, dd))
        words = sum(len(t.split()) for t in ts)
        row['per_seed'].append({'seed': s, 'orig': gppl(ts), 'dedup': gppl(dd), 'words_removed_pct': 100 * removed / words})
        print(name, row['per_seed'][-1], flush=True)
    for k in ('orig', 'dedup', 'words_removed_pct'):
        row[k] = float(np.mean([x[k] for x in row['per_seed']]))
    res[name] = row
    print(name, {k: round(row[k], 3) for k in ('orig', 'dedup', 'words_removed_pct')}, flush=True)
    json.dump(res, open(os.path.expanduser('~/ncm/runs/dedup_test.json'), 'w'), indent=1)
