import argparse
import json
import numpy as np
from pathlib import Path


def inspect_npz(npz_path):
    data = np.load(npz_path)
    print(f"Inspecting {npz_path}")
    print("keys:", list(data.keys()))
    for k in data.keys():
        arr = data[k]
        try:
            print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        except Exception:
            print(f"  {k}: (non-array)")


def cosine_sim(a, b):
    # a, b: 1D
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def analyze(args):
    features_dir = Path(args.features_dir)
    sims = []
    skipped = 0
    processed = 0
    k = max(1, int(getattr(args, "k", 1)))
    with open(args.jsonl, "r") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            vid = obj.get("vid")
            if vid is None:
                continue
            rel_windows = obj.get("relevant_windows") or obj.get("relevant_windows")
            if not rel_windows:
                continue
            npz_name = f"{vid}.npz"
            npz_path = features_dir / npz_name
            if not npz_path.exists():
                alt = npz_name.lstrip(".")
                npz_path = features_dir / alt
                if not npz_path.exists():
                    skipped += 1
                    continue
            data = np.load(npz_path)
            # expect key 'features'
            if "features" in data:
                feats = data["features"]
            else:
                # pick the first array
                k = list(data.keys())[0]
                feats = data[k]

            T = feats.shape[0]
            for w in rel_windows:
                try:
                    s, e = int(w[0]), int(w[1])
                except Exception:
                    continue
                # start boundary (s): inside = mean(frames s .. s+k-1), outside = mean(frames s-k .. s-1)
                in_start = s
                in_end = s + k - 1
                out_start = s - k
                out_end = s - 1
                if in_start <= in_end and out_start <= out_end:
                    in_s = max(0, in_start)
                    in_e = min(T - 1, in_end)
                    out_s = max(0, out_start)
                    out_e = min(T - 1, out_end)
                    if in_s <= in_e and out_s <= out_e:
                        inside_vec = feats[in_s: in_e + 1].mean(axis=0)
                        outside_vec = feats[out_s: out_e + 1].mean(axis=0)
                        sims.append(cosine_sim(inside_vec, outside_vec))

                # end boundary (e): inside = mean(frames e-k .. e-1), outside = mean(frames e .. e+k-1)
                in_start = e - k
                in_end = e - 1
                out_start = e
                out_end = e + k - 1
                if in_start <= in_end and out_start <= out_end:
                    in_s = max(0, in_start)
                    in_e = min(T - 1, in_end)
                    out_s = max(0, out_start)
                    out_e = min(T - 1, out_end)
                    if in_s <= in_e and out_s <= out_e:
                        inside_vec = feats[in_s: in_e + 1].mean(axis=0)
                        outside_vec = feats[out_s: out_e + 1].mean(axis=0)
                        sims.append(cosine_sim(inside_vec, outside_vec))

            processed += 1
            if args.sample_limit and processed >= args.sample_limit:
                break

    sims = np.array(sims, dtype=float)
    out = {"n_pairs": int(sims.size), "mean": float(np.nanmean(sims)) if sims.size else None,
           "std": float(np.nanstd(sims)) if sims.size else None}
    print("Summary:", out)
    # Build a catalog of available feature files (for random sampling)
    catalog = {}
    with open(args.jsonl, 'r') as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            vid = obj.get('vid')
            if vid is None:
                continue
            npz_name = f"{vid}.npz"
            npz_path = Path(args.features_dir) / npz_name
            if not npz_path.exists():
                alt = npz_name.lstrip('.')
                npz_path = Path(args.features_dir) / alt
                if not npz_path.exists():
                    continue
            try:
                data = np.load(npz_path)
                feats = data['features'] if 'features' in data else data[list(data.keys())[0]]
                catalog[vid] = (npz_path, feats.shape[0])
            except Exception:
                continue

    sims_random_same = np.array([], dtype=float)
    sims_random_cross = np.array([], dtype=float)
    if args.n_random is None:
        n_random = sims.size
    else:
        n_random = int(args.n_random)

    if n_random > 0 and len(catalog) > 0:
        import random
        random.seed(args.random_seed)
        # caching loaded feature arrays to avoid repeated IO
        feat_cache = {}

        def load_feats(vid, path):
            if vid in feat_cache:
                return feat_cache[vid]
            data = np.load(path)
            feats = data['features'] if 'features' in data else data[list(data.keys())[0]]
            feat_cache[vid] = feats
            return feats

        vids = list(catalog.keys())
        # same-file random pairs
        same_sims = []
        for _ in range(n_random):
            vid = random.choice(vids)
            path, T = catalog[vid]
            if T < 2:
                continue
            t1 = random.randrange(0, T)
            t2 = random.randrange(0, T)
            if t1 == t2:
                # try to pick different
                t2 = (t1 + 1) % T
            feats = load_feats(vid, path)
            same_sims.append(cosine_sim(feats[t1], feats[t2]))
        sims_random_same = np.array(same_sims, dtype=float)

        # cross-file random pairs
        cross_sims = []
        for _ in range(n_random):
            v1, v2 = random.sample(vids, 2) if len(vids) >= 2 else (vids[0], vids[0])
            p1, T1 = catalog[v1]
            p2, T2 = catalog[v2]
            if T1 < 1 or T2 < 1:
                continue
            t1 = random.randrange(0, T1)
            t2 = random.randrange(0, T2)
            f1 = load_feats(v1, p1)
            f2 = load_feats(v2, p2)
            cross_sims.append(cosine_sim(f1[t1], f2[t2]))
        sims_random_cross = np.array(cross_sims, dtype=float)

    random_summary = {
        'n_random': int(n_random),
        'same_mean': float(np.nanmean(sims_random_same)) if sims_random_same.size else None,
        'same_std': float(np.nanstd(sims_random_same)) if sims_random_same.size else None,
        'cross_mean': float(np.nanmean(sims_random_cross)) if sims_random_cross.size else None,
        'cross_std': float(np.nanstd(sims_random_cross)) if sims_random_cross.size else None,
    }

    # statistical tests (try scipy, otherwise skip)
    stats_res = {}
    try:
        from scipy import stats
        if sims.size and sims_random_same.size:
            t_res = stats.ttest_ind(sims, sims_random_same, equal_var=False, nan_policy='omit')
            mw_res = stats.mannwhitneyu(sims, sims_random_same, alternative='two-sided')
            stats_res['boundary_vs_same_t'] = {'statistic': float(t_res.statistic), 'pvalue': float(t_res.pvalue)}
            stats_res['boundary_vs_same_mw'] = {'statistic': float(mw_res.statistic), 'pvalue': float(mw_res.pvalue)}
        if sims.size and sims_random_cross.size:
            t_res = stats.ttest_ind(sims, sims_random_cross, equal_var=False, nan_policy='omit')
            mw_res = stats.mannwhitneyu(sims, sims_random_cross, alternative='two-sided')
            stats_res['boundary_vs_cross_t'] = {'statistic': float(t_res.statistic), 'pvalue': float(t_res.pvalue)}
            stats_res['boundary_vs_cross_mw'] = {'statistic': float(mw_res.statistic), 'pvalue': float(mw_res.pvalue)}
    except Exception:
        print('scipy not available; skipping statistical tests (install scipy to enable)')

    if args.output:
        np.savez_compressed(args.output, sims=sims, summary=out,
                            sims_random_same=sims_random_same, sims_random_cross=sims_random_cross,
                            random_summary=random_summary, stats=stats_res)
        print(f"Saved results to {args.output}.npz")

    # optional plotting
    try:
        import matplotlib.pyplot as plt
        if sims.size:
            plt.figure(figsize=(6,4))
            bins = 50
            plt.hist(sims, bins=bins, alpha=0.6, label='boundary')
            if sims_random_same.size:
                plt.hist(sims_random_same, bins=bins, alpha=0.4, label='random_same')
            if sims_random_cross.size:
                plt.hist(sims_random_cross, bins=bins, alpha=0.4, label='random_cross')
            plt.title('Cosine similarity: inside vs outside (boundary)')
            plt.xlabel('cosine')
            plt.ylabel('count')
            plt.legend()
            if args.plot:
                plt.savefig(args.plot, bbox_inches='tight')
                print(f"Saved plot to {args.plot}")
    except Exception:
        if args.plot:
            print("matplotlib not available; skipping plot")




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="data/castella_train_release.jsonl")
    p.add_argument("--features-dir", default="features/castella/clap")
    p.add_argument("--sample-limit", type=int, default=0,
                   help="number of JSONL items to process (0 = all)")
    p.add_argument("--k", type=int, default=1,
                   help="half-window size in seconds for averaging (k=1 -> single-frame)")
    p.add_argument("--output", default="src/analysis/results/boundary_sims")
    p.add_argument("--plot", default="src/analysis/results/boundary_sims_hist.png")
    p.add_argument("--n-random", type=int, default=None,
                   help="number of random pairs to sample for baselines (default = n_boundary_pairs)")
    p.add_argument("--random-seed", type=int, default=0,
                   help="random seed for baseline sampling")
    args = p.parse_args()

    Path("src/analysis/results").mkdir(parents=True, exist_ok=True)
    # run analysis
    analyze(args)


if __name__ == "__main__":
    main()
