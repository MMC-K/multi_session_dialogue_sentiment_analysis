import json
import pandas as pd
from pathlib import Path

JSON_PATH = "data.json"   
CSV_PATH  = "label.csv"    
OUT_PATH  = "converted.csv"

def load_dialogs(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dialogs = []
    if isinstance(raw, dict) and "dialog" in raw and "id" in raw:
        dialogs = [raw]  
    elif isinstance(raw, list):
        dialogs = raw    
    else:
        raise ValueError("지원하지 않는 JSON 구조입니다.")

    by_id = {}
    for d in dialogs:
        did = d.get("id")
        seq = []
        for turn in d.get("dialog", []):
            if not isinstance(turn, dict):
                continue
            msg = turn.get("msg")
            speaker = turn.get("speaker") 
            if not isinstance(msg, dict):
                continue
            text = msg.get("text")
            if not text or not isinstance(text, str):
                continue
            if speaker is None:
                continue
            spk = speaker.strip().upper()
            if spk not in ("USER", "SYSTEM"):
                continue
            seq.append({"speaker": spk, "text": text})
        by_id[did] = seq
    return by_id

def build_user_index_map(utter_seq):
    mapping = {}
    ucount = 0
    for i, ut in enumerate(utter_seq):
        if ut["speaker"] == "USER":
            mapping[ucount] = i
            ucount += 1
    return mapping, ucount  

def format_context(utter_seq, pivot_idx, max_prev=4):
    start = max(0, pivot_idx - max_prev)
    ctx = utter_seq[start:pivot_idx+1]
    return "\n".join([f'{u["speaker"]}: {u["text"]}' for u in ctx])

def main(json_path, csv_path, out_path):
    dialogs_by_id = load_dialogs(json_path)
    df = pd.read_csv(csv_path)

    cache = {}  # dialog_id -> (utter_seq, user_map, user_count)

    new_texts = []
    warnings = []

    for i, row in df.iterrows():
        vid = row.get("video_id")
        cid = row.get("clip_id")

        new_text = row.get("text")

        if pd.isna(vid) or pd.isna(cid):
            new_texts.append(new_text)
            continue

        utter_seq = None
        user_map = None
        if vid in cache:
            utter_seq, user_map, user_cnt = cache[vid]
        else:
            utter_seq = dialogs_by_id.get(vid)
            if utter_seq is None:
                warnings.append(f" video_id {vid} 를 JSON에서 찾을 수 없음. 기존 text 유지.")
                new_texts.append(new_text)
                continue
            user_map, user_cnt = build_user_index_map(utter_seq)
            cache[vid] = (utter_seq, user_map, user_cnt)

        if cid not in user_map:
            warnings.append(f" video_id {vid} 의 clip_id {cid} 에 해당하는 USER 발화를 찾을 수 없음. (USER 수={len(user_map)})")
            new_texts.append(new_text)
            continue

        pivot_idx = user_map[cid]
        formatted = format_context(utter_seq, pivot_idx, max_prev=4)
        new_texts.append(formatted)

    df["text"] = new_texts
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 변환 완료: {out_path}")
    if warnings:
        show = "\n".join(warnings[:10])
        more = f"\n... (총 {len(warnings)}개 경고)" if len(warnings) > 10 else ""
        print(show + more)

if __name__ == "__main__":
    main(JSON_PATH, CSV_PATH, OUT_PATH)
