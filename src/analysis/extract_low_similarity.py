import argparse
import json
import numpy as np
from pathlib import Path


def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="data/castella_train_release.jsonl")
    p.add_argument("--features-dir", default="features/castella/clap")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--sample-limit", type=int, default=0,
                   help="0 = all")
    p.add_argument("--top-n", type=int, default=100,
                   help="number of lowest-similarity pairs to save")
    p.add_argument("--output", default="src/analysis/results/low_similarity_pairs.jsonl")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    records = []
    processed = 0
    features_dir = Path(args.features_dir)

    with open(args.jsonl, 'r') as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            vid = obj.get('vid')
            if vid is None:
                continue
            npz_name = f"{vid}.npz"
            npz_path = features_dir / npz_name
            if not npz_path.exists():
                alt = npz_name.lstrip('.')
                npz_path = features_dir / alt
                if not npz_path.exists():
                    continue
            data = np.load(npz_path)
            feats = data['features'] if 'features' in data else data[list(data.keys())[0]]
            T = feats.shape[0]
            rel_windows = obj.get('relevant_windows') or []
            for w in rel_windows:
                try:
                    s, e = int(w[0]), int(w[1])
                except Exception:
                    continue
                # start boundary
                in_start = s
                in_end = s + args.k - 1
                out_start = s - args.k
                out_end = s - 1
                if in_start <= in_end and out_start <= out_end:
                    in_s = max(0, in_start)
                    in_e = min(T - 1, in_end)
                    out_s = max(0, out_start)
                    out_e = min(T - 1, out_end)
                    if in_s <= in_e and out_s <= out_e:
                        inside_vec = feats[in_s: in_e + 1].mean(axis=0)
                        outside_vec = feats[out_s: out_e + 1].mean(axis=0)
                        sim = cosine_sim(inside_vec, outside_vec)
                        records.append({'vid': vid, 'window': [s, e], 'boundary': 'start',
                                        'in_range': [in_s, in_e], 'out_range': [out_s, out_e],
                                        'sim': sim, 'meta': obj})

                # end boundary
                in_start = e - args.k
                in_end = e - 1
                out_start = e
                out_end = e + args.k - 1
                if in_start <= in_end and out_start <= out_end:
                    in_s = max(0, in_start)
                    in_e = min(T - 1, in_end)
                    out_s = max(0, out_start)
                    out_e = min(T - 1, out_end)
                    if in_s <= in_e and out_s <= out_e:
                        inside_vec = feats[in_s: in_e + 1].mean(axis=0)
                        outside_vec = feats[out_s: out_e + 1].mean(axis=0)
                        sim = cosine_sim(inside_vec, outside_vec)
                        records.append({'vid': vid, 'window': [s, e], 'boundary': 'end',
                                        'in_range': [in_s, in_e], 'out_range': [out_s, out_e],
                                        'sim': sim, 'meta': obj})

            processed += 1
            if args.sample_limit and processed >= args.sample_limit:
                break

    if not records:
        print('No records found')
        return

    records.sort(key=lambda x: x['sim'])
    topn = records[:args.top_n]

    # save JSONL
    with open(args.output, 'w') as outfh:
        for r in topn:
            outfh.write(json.dumps(r) + '\n')

    print(f"Saved {len(topn)} low-similarity pairs to {args.output}")
    # print a brief view of the queries/meta for the first 20
    print('\nPreview of queries/meta for lowest 20:')
    for r in topn[:20]:
        meta = r.get('meta', {})
        # print vid and relevant_windows and any textual fields if present
        line = {'vid': r['vid'], 'window': r['window'], 'boundary': r['boundary'], 'sim': r['sim']}
        # show keys of meta and any short text fields (like 'caption' or 'query')
        text_fields = {k: meta.get(k) for k in ['caption', 'query', 'description'] if k in meta}
        if text_fields:
            line['texts'] = text_fields
        print(json.dumps(line))


if __name__ == '__main__':
    main()
