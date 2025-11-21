# check_shards.py
import pickle, os, gc

manifest = "Processed/unaligned/manifest_text.txt"

bad, good = [], []

with open(manifest) as f:
    shards = [line.strip() for line in f if line.strip()]

for i, rel in enumerate(shards, 1):
    path = os.path.join("Processed/unaligned", rel)
    obj = None 
    try:
        with open(path, "rb") as fp:
            obj = pickle.load(fp)
        good.append(path)
        print(f"[GOOD] {path}")
    except Exception as e:
        bad.append((path, repr(e)))
        print(f"[BAD]  {path} -> {e}")
    finally:
        obj = None   # 참조 해제
        gc.collect() 

print(f"\nGood: {len(good)}, Bad: {len(bad)}")

if bad:
    with open("bad_shards.txt", "w") as bf:
        for p, err in bad:
            bf.write(f"{p}\t{err}\n")
    print("=> bad_shards.txt 로 저장됨")
