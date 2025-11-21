import os, json, hashlib, argparse, time, logging  
import requests, cv2, numpy as np, pandas as pd  
from tqdm import tqdm  

SENT2VAL = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}  

# Logging 설정
def setup_logger(verbosity: int):
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG) 
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    return logging.getLogger("build_dataset") 
# IO utils
def read_dialogs(path, root_key=None, logger=None):
    items = [] 
    if logger: logger.info(f"Load JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048); f.seek(0)  
        if head.strip().startswith("{") or head.strip().startswith("["):
            data = json.load(f)  
            if root_key and isinstance(data, dict) and root_key in data and isinstance(data[root_key], list):
                items.extend(data[root_key])  
            elif isinstance(data, list):
                items.extend(data)  
            else:
                items.append(data)  
        else:
            for line in f:
                line=line.strip()
                if line:
                    items.append(json.loads(line))  
    if logger: logger.info(f"Loaded {len(items)} dialog items")
    return items  

def split_mode(key, ratios=(0.8,0.1,0.1)):
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16)  
    r = (h % 10) / 10.0  
    return "train" if r < ratios[0] else ("valid" if r < ratios[0]+ratios[1] else "test")  

def ensure_dir(p): os.makedirs(p, exist_ok=True)  

def img_to_mp4(img_bgr, out_path, fps=25, seconds=1, logger=None):
    h, w = img_bgr.shape[:2]  # 프레임 크기
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))  
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter open failed: {out_path}")  
    frames = max(1, int(fps*seconds))  
    for _ in range(frames): vw.write(img_bgr) 
    vw.release()
    if logger: logger.debug(f"wrote mp4: {out_path}")

def url_to_bgr(url, logger=None):
    if logger: logger.debug(f"GET {url}")
    r = requests.get(url, timeout=20)  
    r.raise_for_status()
    arr = np.frombuffer(r.content, np.uint8)  
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  
    if img is None:
        raise RuntimeError("decode fail: "+url) 
    return img

def blank_bgr(w=640, h=360):
    return np.zeros((h, w, 3), dtype=np.uint8)  


# 메인 빌드
def build(json_path, out_root="YourDataset", fps=25, seconds=1.0, image_policy="reuse_last",
          root_key=None, verbosity=1, log_every=200):
    logger = setup_logger(verbosity)  

    t0 = time.time()  
    ensure_dir(out_root) 
    raw_root = os.path.join(out_root, "Raw"); ensure_dir(raw_root)  

    dialogs = read_dialogs(json_path, root_key=root_key, logger=logger) 
    rows = []  

    stats = {
        "dialogs": len(dialogs),
        "samples_written": 0,
        "user_turns_seen": 0,
        "skipped_no_sentiment": 0,
        "skipped_empty_text": 0,
        "blank_used": 0,
        "img_download_ok": 0,
        "img_download_fail": 0,
    }

    logger.info(f"Start build → out_root={out_root}, fps={fps}, sec={seconds}, policy={image_policy}")  

    for dlg in tqdm(dialogs, desc="Dialogs", unit="dlg"):  
        if not isinstance(dlg, dict):  
            if isinstance(dlg, list):  
                for sub in dlg:
                    if isinstance(sub, dict):
                        _process_one_dialog(sub, raw_root, rows, stats, fps, seconds, image_policy, logger)  
            continue 
        _process_one_dialog(dlg, raw_root, rows, stats, fps, seconds, image_policy, logger)

        if stats["samples_written"] % log_every == 0 and stats["samples_written"] > 0:
            logger.info(f"progress: samples={stats['samples_written']}  "
                        f"(img_ok={stats['img_download_ok']}, img_fail={stats['img_download_fail']}, "
                        f"blank={stats['blank_used']}, skip_no_sent={stats['skipped_no_sentiment']}, "
                        f"skip_empty={stats['skipped_empty_text']})")  
    cols = ["video_id","clip_id","text","label","label_T","label_A","label_V","annotation","mode"] 
    df = pd.DataFrame(rows, columns=cols)  
    label_path = os.path.join(out_root, "label.csv")  
    df.to_csv(label_path, index=False, encoding="utf-8")  

    elapsed = time.time() - t0  
    logger.info(f"[DONE] wrote {len(df)} rows -> {label_path}")  
    logger.info(f"[DONE] Raw videos under {raw_root}") 
    logger.info("Summary: " + ", ".join([f"{k}={v}" for k,v in stats.items()])) 
    logger.info(f"Elapsed: {elapsed:.1f}s") 


def _process_one_dialog(dlg, raw_root, rows, stats, fps, seconds, image_policy, logger):
    video_id = str(dlg.get("id") or hashlib.md5(json.dumps(dlg, ensure_ascii=False).encode()).hexdigest()[:8])  
    vid_dir = os.path.join(raw_root, video_id); ensure_dir(vid_dir)  

    dialog = dlg.get("dialog", []) 
    clip_id = 0  
    last_img_bgr = None  

    for turn in dialog:
        mt = turn.get("msg_type")  
        sp = turn.get("speaker")
        msg = turn.get("msg")  

        if sp == "system" and mt == "image" and isinstance(msg, dict):
            imgs = msg.get("images") or []  
            if imgs:
                src = imgs[0].get("src") 
                if src:
                    try:
                        last_img_bgr = url_to_bgr(src, logger=logger)
                        stats["img_download_ok"] += 1 
                    except Exception as e:
                        logger.debug(f"image download failed: {e}") 
                        stats["img_download_fail"] += 1  
                        last_img_bgr = None 
            continue 

        if sp == "user" and mt == "text" and isinstance(msg, dict):
            stats["user_turns_seen"] += 1  
            ann = msg.get("annotation") or {}  
            sent = ann.get("sentiment")  
            text = (msg.get("text") or "").strip() 

            if not text:
                stats["skipped_empty_text"] += 1 
                continue
            if sent not in SENT2VAL:
                stats["skipped_no_sentiment"] += 1  
                continue

            if image_policy == "reuse_last" and last_img_bgr is not None:
                frame = last_img_bgr 
            elif image_policy == "blank":
                frame = blank_bgr()  
                stats["blank_used"] += 1
            else:
                frame = blank_bgr()  
                stats["blank_used"] += 1

            out_mp4 = os.path.join(vid_dir, f"{clip_id}.mp4")  
            img_to_mp4(frame, out_mp4, fps=fps, seconds=seconds, logger=logger)  

            rows.append({
                "video_id": video_id,
                "clip_id": clip_id,
                "text": text,
                "label": SENT2VAL[sent],
                "label_T": "",  # (미사용)
                "label_A": "",  # (미사용)
                "label_V": "",  # (미사용)
                "annotation": sent.capitalize(),  
                "mode": split_mode(video_id),  
            })
            clip_id += 1  
            stats["samples_written"] += 1 


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)  
    ap.add_argument("--out", default="YourDataset")  
    ap.add_argument("--fps", type=int, default=25)  
    ap.add_argument("--sec", type=float, default=1.0) 
    ap.add_argument("--image_policy", choices=["reuse_last","blank"], default="reuse_last")  
    ap.add_argument("--root_key", default=None, help="최상위가 {'data':[...]} 형태일 때 key 지정")  
    ap.add_argument("--verbosity", type=int, default=1, choices=[0,1,2], help="0=warn, 1=info, 2=debug")  
    ap.add_argument("--log_every", type=int, default=200, help="n개 샘플마다 진행 로그") 
    args = ap.parse_args()

    build(
        args.json, out_root=args.out, fps=args.fps, seconds=args.sec,
        image_policy=args.image_policy, root_key=args.root_key,
        verbosity=args.verbosity, log_every=args.log_every
    ) 