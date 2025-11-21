import os, sys, pickle, argparse
from pathlib import Path
import numpy as np

def map_reg_to_cls(v, eps=1e-6):
    if abs(v - (-1.0)) <= eps: return 0
    if abs(v - 0.0)   <= eps: return 1
    if abs(v - 1.0)   <= eps: return 2
    raise ValueError(f"Unexpected regression label value: {v}")

NEEDED_IN = ['id','raw_text','text','text_bert','annotations','regression_labels']
NEEDED_OUT = ['raw_text','audio','vision','id','text','text_bert',
              'audio_lengths','vision_lengths','annotations',
              'classification_labels','regression_labels']

def convert_split(split_dict, shard_path):
    for k in NEEDED_IN:
        if k not in split_dict:
            raise KeyError(f"[{shard_path}] split lacks key: '{k}'")
    n = len(split_dict['id'])
    for k in NEEDED_IN:
        if len(split_dict[k]) != n:
            raise ValueError(f"[{shard_path}] length mismatch: key '{k}' has {len(split_dict[k])} vs id {n}")

    out = {k: split_dict[k] for k in ['id','raw_text','text','text_bert','annotations','regression_labels']}

    out['audio'] = [np.zeros((0,), dtype=np.float32) for _ in range(n)]
    out['vision'] = [np.zeros((0,), dtype=np.float32) for _ in range(n)]
    out['audio_lengths'] = [0]*n
    out['vision_lengths'] = [0]*n

    reg = split_dict['regression_labels']
    reg_iter = reg.tolist() if hasattr(reg, 'tolist') else reg
    out['classification_labels'] = [map_reg_to_cls(float(x)) for x in reg_iter]

    ordered = {k: out[k] for k in NEEDED_OUT}
    return ordered

def convert_shard(in_path, out_path):
    with open(in_path, 'rb') as f:
        data = pickle.load(f) 

    new_data = {}
    for split in ['train','valid','test']:
        if split not in data:
            raise KeyError(f"[{in_path}] missing split: {split}")
        new_data[split] = convert_split(data[split], in_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    ap = argparse.ArgumentParser(description="Convert Ft shards to MMSA Ft-only shards and build new manifest.")
    ap.add_argument("--manifest", required=True,
                    help="원본 manifest 상대경로")
    ap.add_argument("--out_dir", required=True, help="변환된 pkl 샤드를 저장할 디렉터리")
    ap.add_argument("--suffix", default="_ftonly", help="출력 파일명에 붙일 접미사")
    ap.add_argument("--out_manifest", default=None, help="새 manifest 경로 (기본: out_dir/manifest_ftonly.txt)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    base_dir = manifest_path.parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_manifest = Path(args.out_manifest) if args.out_manifest else (out_dir / "manifest_ftonly.txt")

    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    new_lines = []
    for i, rel in enumerate(lines, 1):
        in_path = (base_dir / rel).resolve() if not os.path.isabs(rel) else Path(rel)
        if not in_path.exists():
            print(f"[WARN] Missing shard: {in_path}", file=sys.stderr)
            continue

        stem = in_path.stem  # ourdata_Vt_unaligned.pkl.part00001_1758174125.pkl -> 'ourdata_Vt_unaligned.pkl.part00001_1758174125_ftonly.pkl'
        if stem.endswith(".pkl"):
            out_name = stem[:-4] + args.suffix + ".pkl"
        else:
            out_name = stem + args.suffix + ".pkl"
        out_path = out_dir / out_name

        print(f"[{i}/{len(lines)}] {in_path.name} -> {out_path.name}")
        convert_shard(in_path, out_path)

        new_lines.append(out_path.relative_to(out_dir).as_posix())

    with open(out_manifest, "w", encoding="utf-8") as f:
        for ln in new_lines:
            f.write(ln + "\n")

    print(f"\nDone. Wrote {len(new_lines)} shards.")
    print(f"New manifest: {out_manifest}")
    print(f"Output dir   : {out_dir}")

if __name__ == "__main__":
    main()
