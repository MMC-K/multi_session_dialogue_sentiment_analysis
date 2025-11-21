# pkl 파일과 manifest 파일을 제작하는 코드

import os, csv, random, pickle, tempfile, shutil, json, gc, sys, glob
import multiprocessing as mp
from typing import List

import absl.logging

THIS_DIR = os.path.abspath(os.path.dirname(__file__))                  #MMSA_master/MMSA-FET
SRC_ROOT = os.path.join(THIS_DIR, "src")                               # MMSA_master/MMSA-FET/src

if os.path.isdir(SRC_ROOT) and SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

os.environ["PYTHONPATH"] = SRC_ROOT + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


from MSA_FET.dataset import run_dataset

CFG = os.environ.get(
    "CFG_PATH",
    "MMSA_master/MMSA-FET/config_text_video.json"
)
OUT = "ourdata_Vt_unaligned.pkl"

def read_rows(label_csv: str):
    with open(label_csv, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]

def maybe_patch_video_args_for_extractor(cfg_path: str) -> str:
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "args" in cfg:
            return cfg_path  
        vargs = (cfg.get("video") or {}).get("args")
        if not isinstance(vargs, dict):
            return cfg_path  

        if "visualize_dir" in vargs:
            vargs["visualize_dir"] = os.path.expanduser(vargs["visualize_dir"])

        cfg["args"] = vargs

        tmp = tempfile.mkdtemp(prefix="cfg_patch_videoargs_")
        tmp_cfg = os.path.join(tmp, "config_text_video_patched.json")
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return tmp_cfg
    except Exception:
        return cfg_path

def write_label(dst_dir: str, header: List[str], rows: List[List[str]]):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "label.csv")
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def iter_chunks(data: List[List[str]], shard_size: int):
    for i in range(0, len(data), shard_size):
        yield i // shard_size, data[i:i+shard_size]

def safe_int_env(name: str, default: int) -> int:
    try: return int(os.environ.get(name, str(default)))
    except Exception: return default

def maybe_patch_batchsize(cfg_path: str) -> str:
    bs = os.environ.get("BATCH_SIZE", "").strip()
    if not bs: return cfg_path
    try: bs_val = int(bs)
    except Exception: return cfg_path
    if bs_val < 1: return cfg_path
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "text" in cfg and "args" in cfg["text"]:
            cur = cfg["text"]["args"].get("batch_size", None)
            if cur is None or cur > bs_val:
                cfg["text"]["args"]["batch_size"] = bs_val
            tmp = tempfile.mkdtemp(prefix="cfg_patch_")
            tmp_cfg = os.path.join(tmp, "config_text.json")
            with open(tmp_cfg, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            return tmp_cfg
    except Exception:
        pass
    return cfg_path

def make_mini(src_label: str, limit: int):
    if limit <= 0: return None
    mini_dir = f"./mini_{limit}"
    os.makedirs(mini_dir, exist_ok=True)
    dst = os.path.join(mini_dir, "label.csv")
    with open(src_label, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    random.seed(42); random.shuffle(data)
    data = data[:limit]
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(data)
    return mini_dir

def write_manifest_txt(manifest_path: str, part_files: list):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for p in part_files:
            f.write(os.path.basename(p) + "\n")

def is_complete_part(path: str, min_bytes: int) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) >= min_bytes and not path.endswith(".tmp")

def clean_broken_temp(out_glob: str):
    # 지지난/중단된 임시파일 정리
    for p in glob.glob(out_glob + ".tmp"):
        try: os.remove(p)
        except Exception: pass
import os, re, threading

def install_native_stderr_filter():
    drop_line = re.compile(
        rb'^I\d{4}.*(?:gl_context_egl\.cc:\d+|gl_context\.cc:\d+|GL version:|Successfully initialized EGL)\b',
        re.IGNORECASE,
    )

    orig_fd2 = os.dup(2)

    pipe_r, pipe_w = os.pipe()
    os.dup2(pipe_w, 2)
    os.close(pipe_w)

    def _pump():
        buf = b""
        with os.fdopen(pipe_r, "rb", buffering=0) as r, os.fdopen(orig_fd2, "wb", buffering=0) as w:
            while True:
                chunk = r.read(4096)
                if not chunk:
                    if buf and not drop_line.match(buf):
                        w.write(buf)
                        w.flush()
                    break

                parts = chunk.split(b'\r')
                for i, seg in enumerate(parts):
                    if i > 0: 
                        w.write(b'\r')
                    buf += seg
                    while True:
                        nl = buf.find(b'\n')
                        if nl < 0:
                            break
                        line, buf = buf[:nl], buf[nl+1:]
                        if not drop_line.match(line):
                            w.write(line + b'\n')
                    w.flush()
                # 루프 계속

    t = threading.Thread(target=_pump, name="stderr-filter", daemon=True)
    t.start()


def _worker_extract(cfg_path: str, dataset_dir: str, final_part_path: str, workers: int):
    install_native_stderr_filter() 
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

        tmp_path = final_part_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        run_dataset(
            cfg_path,
            dataset_dir=dataset_dir,
            out_file=tmp_path,       
            return_type="list",
            num_workers=max(1, workers),
        )


        if os.path.exists(final_part_path):
            os.remove(final_part_path)
        os.replace(tmp_path, final_part_path)
        return 0
    except Exception as e:
        try:
            if os.path.exists(final_part_path + ".tmp"):
                os.remove(final_part_path + ".tmp")
        except Exception:
            pass
        print(f"[ERROR] worker failed for {final_part_path}: {e}", file=sys.stderr)
        return 1
def link_if_exists(src_dir: str, dst_dir: str):
    if not os.path.isdir(src_dir):
        return
    if os.path.lexists(dst_dir):
        return
    try:
        os.symlink(os.path.abspath(src_dir), dst_dir, target_is_directory=True)
    except Exception:
        import shutil
        shutil.copytree(src_dir, dst_dir, symlinks=True)

def link_modal_dirs(asset_root: str, shard_dir: str):
    for name in ["Raw", "Text", "texts", "Frames", "Images", "Image"]:
        src = os.path.join(asset_root, name)
        dst = os.path.join(shard_dir, name)
        link_if_exists(src, dst)

def main():
    import inspect, importlib
    mp_mod = importlib.import_module("MSA_FET.extractors.video.mediapipe")
    print("[CHECK] mediapipe.py =>", inspect.getfile(mp_mod))

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    workers   = safe_int_env("WORKERS", 1)         
    limit     = safe_int_env("LIMIT",   0)
    shard_sz  = safe_int_env("SHARD", 5000)
    min_bytes = safe_int_env("MIN_PART_BYTES", 200_000_000)  # 완료판정 하한(바이트)

    dataset_dir = "."
    label_csv   = os.path.join(dataset_dir, "label.csv")
    cfg_path    = maybe_patch_batchsize(CFG)
    cfg_path = maybe_patch_video_args_for_extractor(cfg_path)  

    if not os.path.isfile(label_csv):
        raise FileNotFoundError(f"label.csv not found: {label_csv}")

    if limit > 0:
        dataset_dir = make_mini(label_csv, limit) or dataset_dir
        label_csv   = os.path.join(dataset_dir, "label.csv")

    header, data = read_rows(label_csv)
    total = len(data)
    if total == 0:
        print("No rows to process. Exit.")
        return

    out_file = OUT if limit == 0 else OUT.replace(".pkl", f"_limit{limit}.pkl")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    clean_broken_temp(out_file + ".part*")

    print(f"[INFO] rows={total}, shard={shard_sz}, workers(per-proc)={workers}")
    print(f"[INFO] cfg: {cfg_path} (BATCH_SIZE={os.environ.get('BATCH_SIZE','') or 'unchanged'})")
    print(f"[INFO] out prefix: {out_file}")

    tmp_root = tempfile.mkdtemp(prefix="fet_shard_")
    part_files = []
    processed = 0
    try:
        for idx, chunk in iter_chunks(data, shard_sz):
            shard_dir = os.path.join(tmp_root, f"shard_{idx:05d}")
            write_label(shard_dir, header, chunk)
            link_modal_dirs(os.environ.get("ASSET_ROOT","."), shard_dir)

            final_part = out_file + f".part{idx:05d}.pkl"
            part_files.append(final_part)

            if is_complete_part(final_part, min_bytes):
                processed += len(chunk)
                print(f"[RESUME] skip shard {idx} (exists, size={os.path.getsize(final_part)/1e6:.1f} MB) "
                      f"(processed={processed}/{total})")
                shutil.rmtree(shard_dir, ignore_errors=True)
                continue

            proc = mp.Process(target=_worker_extract,
                              args=(cfg_path, shard_dir, final_part, workers))
            proc.start()
            proc.join()

            if proc.exitcode != 0 or not is_complete_part(final_part, min_bytes):
                print(f"shard {idx} failed or incomplete. retry once...")
                try:
                    if os.path.exists(final_part): os.remove(final_part)
                    if os.path.exists(final_part + ".tmp"): os.remove(final_part + ".tmp")
                except Exception:
                    pass
                proc2 = mp.Process(target=_worker_extract,
                                   args=(cfg_path, shard_dir, final_part, workers))
                proc2.start()
                proc2.join()
                if proc2.exitcode != 0 or not is_complete_part(final_part, min_bytes):
                    raise RuntimeError(f"Shard {idx} failed twice.")

            processed += len(chunk)
            print(f"shard {idx} done: {len(chunk)} rows (processed={processed}/{total})")

            try: shutil.rmtree(shard_dir, ignore_errors=True)
            except Exception: pass
            gc.collect()

        completed = [p for p in part_files if is_complete_part(p, min_bytes)]
        manifest_path = os.path.join(os.path.dirname(out_file), "manifest_text.txt")
        write_manifest_txt(manifest_path, completed)
        print(f"[INFO] wrote manifest: {manifest_path} (count={len(completed)})")
        print(f"[INFO] keep shard pkls only (no merge).")

    finally:
        try: shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception: pass

if __name__ == "__main__":
    main()
