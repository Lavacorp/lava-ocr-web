# local_ocr_agent.py — RapidOCR + Auto-rotate + Fast Container + Dual Template + Barcode/QR + OpenAI Fallback
# Python 3.13 호환 / torch 불필요

import os, time, json, base64, mimetypes, re, cv2, numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Tuple, Optional

# ======================= 환경설정 로드 =======================
load_dotenv("config.env")

OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")
USE_DROPBOX  = os.getenv("USE_DROPBOX", "0") == "1"
DROPBOX_TOKEN = os.getenv("DROPBOX_TOKEN", "")
DROPBOX_FOLDER = os.getenv("DROPBOX_FOLDER", "/work/LAVA의 팀 폴더/라바상사/CON_PHOTO").rstrip("/")

if not OPENAI_KEY:
    raise SystemExit("❌ OPENAI_API_KEY가 설정되지 않았습니다. config.env 또는 환경변수 확인!")

from rapidocr_onnxruntime import RapidOCR
client = OpenAI(api_key=OPENAI_KEY)
ocr_engine = RapidOCR()  # 가벼운 OCR 엔진 (ONNXRuntime, CPU)

# ====== 경로/옵션 ======
PHOTOS_DIR = "photos"     # 로컬 감시 루트(Use Dropbox=0인 경우)
OUTPUT_DIR = "output"
DEBUG_DIR  = os.path.join(OUTPUT_DIR, "_debug")
CHECK_INTERVAL = 30       # 초

SAVE_DEBUG = False
DEBUG_SAMPLE_RATE = 0
MAX_LONG_SIDE = 1600
ORIENT_SAMPLE_MAX = 960

# ====== 템플릿 ======
TPL_P = os.path.join(os.path.dirname(__file__), "template_portrait.jpg")
TPL_L = os.path.join(os.path.dirname(__file__), "template_landscape.jpg")
USE_TEMPLATE = os.path.exists(TPL_P) or os.path.exists(TPL_L)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
if not USE_DROPBOX:
    os.makedirs(PHOTOS_DIR, exist_ok=True)

MASTER_XLSX = os.path.join(OUTPUT_DIR, "results.xlsx")
MASTER_CSV  = os.path.join(OUTPUT_DIR, "results.csv")

# ====== (선택) Dropbox ======
dbx = None
TMP_DIR = os.path.join(OUTPUT_DIR, "_tmp_from_dropbox")
if USE_DROPBOX:
    if not DROPBOX_TOKEN:
        raise SystemExit("❌ USE_DROPBOX=1 이지만 DROPBOX_TOKEN 이 없습니다. config.env 확인!")
    import dropbox
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    os.makedirs(TMP_DIR, exist_ok=True)

# ======================= ROI 정의 =======================
TEMPLATE_W, TEMPLATE_H = 1400, 800

ROIS_PORTRAIT = {
    "HEAT_NO":   (140, 420, 420, 80),
    "BUNDLE_NO": (720, 420, 520, 80),
    "SIZE":      (140, 320, 420, 80),
    "GRADE":     (720, 210, 520, 80),
    "LENGTH":    (720, 320, 520, 80),
    "WEIGHT":    (720, 530, 520, 80),
}
ROIS_LANDSCAPE = {
    "HEAT_NO":   (520, 470, 320, 80),
    "BUNDLE_NO": (880, 470, 320, 80),
    "SIZE":      (520, 360, 320, 80),
    "GRADE":     (190, 250, 740, 90),
    "LENGTH":    (880, 360, 320, 80),
    "WEIGHT":    (880, 580, 320, 80),
}

# ======================= 유틸리티 =======================
def is_image_file(n: str) -> bool:
    return n.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"))

def move_to_done_local(path: str):
    try:
        done_dir = os.path.join(os.path.dirname(path), "_done")
        os.makedirs(done_dir, exist_ok=True)
        os.replace(path, os.path.join(done_dir, os.path.basename(path)))
    except Exception as e:
        print(f"[move_to_done_local] {e}")

def move_to_done_dropbox(dbx_path: str):
    # /a/b/c.jpg -> /a/b/_done/c.jpg
    dir_name = os.path.dirname(dbx_path)
    base = os.path.basename(dbx_path)
    done_dir = f"{dir_name}/_done"
    done_path = f"{done_dir}/{base}"
    try:
        dbx.files_move_v2(dbx_path, done_path, autorename=True)
    except Exception:
        try:
            dbx.files_create_folder_v2(done_dir)
        except Exception:
            pass
        try:
            dbx.files_move_v2(dbx_path, done_path, autorename=True)
        except Exception as e2:
            print(f"[move_to_done_dropbox] {e2}")

NORMAL_KEY_MAP = {
    "containernumber":"container_number","container":"container_number",
    "containerno":"container_number","container_no":"container_number","container#":"container_number",
    "heatno":"heat_no","heat_no":"heat_no","heat":"heat_no",
    "bundleno":"bundle_no","bundle_no":"bundle_no","bundle":"bundle_no",
    "size":"size","grade":"grade","length":"length","weight":"weight",
}
def normalize_key(k: str) -> str:
    if not isinstance(k, str): return k
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", k).lower()
    return NORMAL_KEY_MAP.get(cleaned, k)

def normalize_fields(fields: Dict[str, str]) -> Dict[str, str]:
    out = {}
    for k, v in fields.items():
        out[normalize_key(k)] = v
    return out

def all_fields_empty(fields: Dict[str, str]) -> bool:
    if not fields: return True
    return not any((isinstance(v, str) and v.strip()) for v in fields.values())

def resize_long_side(img, max_side):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side: return img
    s = max_side / float(m)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

_CONTAINER_RE = re.compile(r"\b([A-Z]{4}[-\s]?\d{6}[-\s]?\d)\b")
def extract_container_no(text: str) -> Optional[str]:
    if not text: return None
    m = _CONTAINER_RE.search(text.upper().replace("-", " "))
    if not m: return None
    return m.group(1).replace(" ","").replace("-","")

# ======================= 방향 교정 =======================
def rapid_text_len(gray: np.ndarray) -> int:
    res, _ = ocr_engine(gray)
    if not res: return 0
    return sum(len(t[1]) for t in res)

def autorotate_best(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = resize_long_side(img_bgr, ORIENT_SAMPLE_MAX)
    cands = [
        img_bgr,
        cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_bgr, cv2.ROTATE_180),
        cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    best, s_best = img_bgr, -1
    for c in cands:
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        score = rapid_text_len(gray)
        if score > s_best:
            best, s_best = c, score
    return best

# ======================= 템플릿 정합 =======================
def align_to_template(img_bgr: np.ndarray, template_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(img_bgr, None)
    kp2, des2 = orb.detectAndCompute(template_bgr, None)
    if des1 is None or des2 is None: return None, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m,n in matches if n is not None and m.distance < 0.75*n.distance]
    if len(good) < 12: return None, len(good)
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H,_ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None: return None, len(good)
    return cv2.warpPerspective(img_bgr, H, (TEMPLATE_W, TEMPLATE_H)), len(good)

def choose_best_template(img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str], int]:
    cands=[]
    if os.path.exists(TPL_P): cands.append(("portrait", cv2.imread(TPL_P)))
    if os.path.exists(TPL_L): cands.append(("landscape", cv2.imread(TPL_L)))
    best_warp, best_name, best_match = None, None, 0
    for name, tmpl in cands:
        warped, mc = align_to_template(img, tmpl)
        if warped is not None and mc > best_match:
            best_warp, best_name, best_match = warped, name, mc
    return best_warp, best_name, best_match

# ======================= ROI OCR =======================
def enhance_variants(gray: np.ndarray):
    variants=[gray]
    variants.append(cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
    variants.append(cv2.bitwise_not(gray))
    variants.append(cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY,31,5))
    return variants

def rapidocr_text(img: np.ndarray) -> str:
    res, _ = ocr_engine(img)
    if not res: return ""
    return " ".join([t[1] for t in res if t[1]])

def ocr_roi_best(crop_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    best_txt, best_len = "", -1
    for v in enhance_variants(gray):
        txt = rapidocr_text(v)
        if len(txt) > best_len:
            best_txt, best_len = txt, len(txt)
    h,w = gray.shape
    for half in (gray[:, :w//2], gray[:, w//2:]):
        txt = rapidocr_text(half)
        if len(txt) > best_len:
            best_txt, best_len = txt, len(txt)
    return best_txt.strip()

# ======================= 바코드/QR =======================
_QR = cv2.QRCodeDetector()
def decode_qr_numbers(img_bgr: np.ndarray) -> str:
    texts = []
    data, pts, _ = _QR.detectAndDecode(img_bgr)
    if data: texts.append(data)
    if not texts:
        h, w = img_bgr.shape[:2]
        roi = img_bgr[int(h*0.70):h, int(w*0.55):w]
        d2, p2, _ = _QR.detectAndDecode(roi)
        if d2: texts.append(d2)
    nums=[]
    for t in texts:
        nums += re.findall(r"\d+", t)
    return " ".join(nums).strip()

# ======================= 템플릿 기반 (라벨) =======================
def extract_by_template_dual(img_path: str, rel_dbg: str) -> Tuple[Optional[Dict[str,str]], str]:
    if not USE_TEMPLATE:
        return None, "template_disabled"
    try:
        raw = cv2.imread(img_path)
        if raw is None: return None, "imread_failed"
        raw = resize_long_side(raw, MAX_LONG_SIDE)
        img = autorotate_best(raw)

        warped, tpl_name, match_cnt = choose_best_template(img)
        if warped is None:
            return None, f"align_failed({match_cnt})"

        if SAVE_DEBUG and (DEBUG_SAMPLE_RATE<=0 or (hash(rel_dbg) % DEBUG_SAMPLE_RATE==0)):
            dbg_warp = os.path.join(DEBUG_DIR, rel_dbg.replace(os.sep,"__")+f".{tpl_name}.warped.jpg")
            os.makedirs(os.path.dirname(dbg_warp), exist_ok=True)
            cv2.imwrite(dbg_warp, warped)

        rois = ROIS_PORTRAIT if tpl_name=="portrait" else ROIS_LANDSCAPE
        out={}
        for k,(x,y,w,h) in rois.items():
            crop = warped[y:y+h, x:x+w]
            txt  = ocr_roi_best(crop)
            out[normalize_key(k)] = txt
            if SAVE_DEBUG and (DEBUG_SAMPLE_RATE<=0 or (hash(rel_dbg) % DEBUG_SAMPLE_RATE==0)):
                dbg_roi = os.path.join(DEBUG_DIR, rel_dbg.replace(os.sep,"__")+f".{tpl_name}.{k}.jpg")
                cv2.imwrite(dbg_roi, crop)

        # 바코드/QR
        h_total = warped.shape[0]
        barcode_region = warped[int(h_total*0.85):h_total, :]
        barcode_txt = rapidocr_text(cv2.cvtColor(barcode_region, cv2.COLOR_BGR2GRAY))
        barcode_txt = re.sub(r"[^0-9\s]", "", barcode_txt).strip()
        out["barcode"] = barcode_txt
        out["qr_numbers"] = decode_qr_numbers(warped)

        return normalize_fields(out), ""
    except Exception as e:
        return None, f"template_exception:{e}"

# ======================= 컨테이너 전면 OCR =======================
def find_container_by_full_ocr(img_bgr: np.ndarray) -> Optional[str]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    res, _ = ocr_engine(gray)
    if not res: return None
    text = " ".join([t[1] for t in res if t[1]])
    return extract_container_no(text)

# ======================= OpenAI Vision (폴백) =======================
PROMPT = (
    "You are an OCR agent for steel logistics.\n"
    "Detect either a shipping container number (ISO 6346, e.g., ABCD1234567), "
    "or a product label with fields (HEAT NO, BUNDLE NO, SIZE, GRADE, LENGTH, WEIGHT). "
    "Even if field titles are occluded by hands, infer by table positions/layout.\n"
    "Return ONLY JSON like: {\"type\":\"container\"|\"label\",\"fields\":{...}}"
)

def analyze_with_openai(image_path: str) -> Tuple[Dict[str,str], str, str]:
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        mime,_ = mimetypes.guess_type(image_path)
        if not mime: mime = "image/jpeg"
        b64 = base64.b64encode(img_bytes).decode()
        url = f"data:{mime};base64,{b64}"

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role":"user","content":[
                {"type":"text","text":PROMPT},
                {"type":"image_url","image_url":{"url":url}}
            ]}]
        )
        txt = res.choices[0].message.content.strip()
        raw = re.sub(r"^```json|```$", "", txt, flags=re.M).strip()

        try:
            data = json.loads(raw)
        except Exception:
            guess = extract_container_no(raw)
            return normalize_fields({"container_number": guess or ""}), raw, "json_parse_failed"

        typ = (data.get("type") or "").strip().lower()
        fields = data.get("fields", {}) if isinstance(data.get("fields"), dict) else {}
        fields_norm = normalize_fields(fields)
        if typ=="container" and not fields_norm.get("container_number"):
            g = extract_container_no(raw)
            if g: fields_norm["container_number"] = g
        fields_norm["type"] = typ
        return fields_norm, raw, ""
    except Exception as e:
        return {}, "", f"openai_exception:{e}"

# ======================= 저장(폴더별 단일 파일명) =======================
def save_master_and_per_folder(rows: list):
    df = pd.DataFrame(rows)
    for c in ("folder","relpath","file"):
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    preferred = [
        "folder","relpath","file","method",
        "type","container_number",
        "heat_no","bundle_no","size","grade","length","weight",
        "barcode","qr_numbers",
        "raw_text","error"
    ]
    cols = list(df.columns)
    df = df[[c for c in preferred if c in cols] + [c for c in cols if c not in preferred]]

    df.to_csv(MASTER_CSV, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(MASTER_XLSX, index=False)
        print(f"✅ Master saved → {MASTER_XLSX} ({len(df)} rows)")
    except Exception as e:
        print(f"⚠️ Excel 저장 생략: {e}")

    # 폴더별 단일 파일 {폴더명}_bundle list.xlsx
    for folder, g in df.groupby("folder", dropna=False):
        folder_name = (folder or "").strip() or "root"
        sub_output = OUTPUT_DIR
        os.makedirs(sub_output, exist_ok=True)
        out_name = f"{folder_name}_bundle list.xlsx"
        out_path = os.path.join(sub_output, out_name)
        g.to_excel(out_path, index=False)
        print(f"📦 Folder saved → {out_path} ({len(g)} rows)")

# ======================= Dropbox helpers =======================
IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")

def dbx_is_image(path_lower: str) -> bool:
    return path_lower.lower().endswith(IMG_EXTS)

def dbx_list_images(folder_path: str):
    paths = []
    try:
        res = dbx.files_list_folder(folder_path, recursive=True)
        while True:
            for entry in res.entries:
                if isinstance(entry, type(res.entries[0])):  # runtime-safe
                    pass
            for entry in res.entries:
                if hasattr(entry, "path_lower") and hasattr(entry, "name"):
                    # 파일만 대상으로 처리
                    if getattr(entry, "client_modified", None) is not None and dbx_is_image(entry.path_lower):
                        paths.append(entry.path_lower)
            if not res.has_more:
                break
            res = dbx.files_list_folder_continue(res.cursor)
    except Exception as e:
        print("⚠️ Dropbox list error:", e)
    return paths

def dbx_download_to_tmp(dbx_path: str) -> str:
    rel = dbx_path
    if dbx_path.lower().startswith(DROPBOX_FOLDER.lower()+"/"):
        rel = dbx_path[len(DROPBOX_FOLDER):].lstrip("/")
    local_dir = os.path.join(TMP_DIR, os.path.dirname(rel))
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(dbx_path))
    try:
        md, resp = dbx.files_download(dbx_path)
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    except Exception as e:
        print("⚠️ Dropbox download error:", e)
        return ""

# ======================= 메인 루프 =======================
def main():
    print(f"🚀 OCR Agent | USE_TEMPLATE={USE_TEMPLATE} | USE_DROPBOX={USE_DROPBOX}")
    results, seen = [], set()

    # 이전 결과 로드
    if os.path.exists(MASTER_CSV):
        try:
            prev = pd.read_csv(MASTER_CSV, dtype=str).fillna("")
            results = prev.to_dict("records")
            seen = {r.get("relpath","") for r in results}
            print(f"ℹ️ Loaded previous results: {len(results)} rows")
        except Exception as e:
            print(f"⚠️ 기존 결과 불러오기 실패: {e}")

    while True:
        try:
            new_found = 0

            if USE_DROPBOX:
                # Dropbox 모드
                dbx_paths = dbx_list_images(DROPBOX_FOLDER)
                for dbx_path in dbx_paths:
                    rel = dbx_path  # 고유 key로 사용
                    if rel in seen:
                        continue
                    local_path = dbx_download_to_tmp(dbx_path)
                    if not local_path:
                        continue

                    name = os.path.basename(local_path)
                    print(f"📸 [DBX] {dbx_path} 분석 중 ...")
                    folder_for_output = os.path.dirname(rel).replace(DROPBOX_FOLDER, "").strip("/") or "root"

                    try:
                        raw0 = cv2.imread(local_path)
                        if raw0 is None:
                            raise RuntimeError("imread_failed")
                        raw0 = resize_long_side(raw0, MAX_LONG_SIDE)
                        raw0 = autorotate_best(raw0)

                        cont = find_container_by_full_ocr(raw0)
                        if cont:
                            row = {
                                "folder": folder_for_output,
                                "relpath": rel,
                                "file": name,
                                "method": "fullocr",
                                "type": "container",
                                "container_number": cont,
                                "error": "",
                                "raw_text": json.dumps({"container_number": cont}, ensure_ascii=False)
                            }
                            results.append(row)
                            save_master_and_per_folder(results)
                            seen.add(rel)
                            move_to_done_dropbox(dbx_path)
                            new_found += 1
                            continue

                        fields, method, err, raw_text = {}, "template" if USE_TEMPLATE else "openai", "", ""
                        if USE_TEMPLATE:
                            fields, err = extract_by_template_dual(local_path, rel)
                            if not isinstance(fields, dict):
                                fields = {}
                            key_fields = ["heat_no","bundle_no","size","grade","length","weight"]
                            non_empty = sum(1 for k in key_fields if str(fields.get(k,"")).strip())
                            if not fields or non_empty <= 1:
                                method = "openai"

                        if method == "openai":
                            fields2, raw_text2, err2 = analyze_with_openai(local_path)
                            if not isinstance(fields2, dict): fields2 = {}
                            fields = fields2; raw_text = raw_text2; err = f"{err};{err2}".strip(";")

                        row = {
                            "folder": folder_for_output,
                            "relpath": rel,
                            "file": name,
                            "method": method,
                            "error": err,
                            "raw_text": raw_text if raw_text else json.dumps(fields, ensure_ascii=False),
                        }
                        for k, v in (fields or {}).items():
                            row[normalize_key(k)] = v
                        results.append(row)
                        save_master_and_per_folder(results)
                        seen.add(rel)
                        move_to_done_dropbox(dbx_path)
                        new_found += 1

                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        results.append({
                            "folder": folder_for_output,
                            "relpath": rel,
                            "file": name,
                            "method": "error",
                            "error": str(e),
                            "raw_text": "",
                        })
                        save_master_and_per_folder(results)
                        seen.add(rel)
                        print(f"⚠️ 파일 처리 오류({rel}): {e}")

            else:
                # 로컬 모드
                for root, dirs, files in os.walk(PHOTOS_DIR):
                    dirs[:] = [d for d in dirs if not d.startswith("_")]
                    for name in files:
                        if not is_image_file(name):
                            continue
                        full = os.path.join(root, name)
                        rel  = os.path.relpath(full, PHOTOS_DIR)
                        if rel in seen:
                            continue

                        print(f"📸 {rel} 분석 중 ...")
                        try:
                            raw0 = cv2.imread(full)
                            if raw0 is None: raise RuntimeError("imread_failed")
                            raw0 = resize_long_side(raw0, MAX_LONG_SIDE)
                            raw0 = autorotate_best(raw0)

                            cont = find_container_by_full_ocr(raw0)
                            if cont:
                                row = {
                                    "folder": os.path.dirname(rel) or "root",
                                    "relpath": rel,
                                    "file": name,
                                    "method": "fullocr",
                                    "type": "container",
                                    "container_number": cont,
                                    "error": "",
                                    "raw_text": json.dumps({"container_number": cont}, ensure_ascii=False)
                                }
                                results.append(row)
                                save_master_and_per_folder(results)
                                seen.add(rel); new_found += 1
                                move_to_done_local(full)
                                continue

                            fields, method, err, raw_text = {}, "template" if USE_TEMPLATE else "openai", "", ""
                            if USE_TEMPLATE:
                                fields, err = extract_by_template_dual(full, rel)
                                if not isinstance(fields, dict): fields = {}
                                key_fields = ["heat_no","bundle_no","size","grade","length","weight"]
                                non_empty = sum(1 for k in key_fields if str(fields.get(k,"")).strip())
                                if not fields or non_empty <= 1:
                                    method = "openai"

                            if method == "openai":
                                fields2, raw_text2, err2 = analyze_with_openai(full)
                                if not isinstance(fields2, dict): fields2 = {}
                                fields = fields2; raw_text = raw_text2; err = f"{err};{err2}".strip(";")

                            row = {
                                "folder": os.path.dirname(rel) or "root",
                                "relpath": rel,
                                "file": name,
                                "method": method,
                                "error": err,
                                "raw_text": raw_text if raw_text else json.dumps(fields, ensure_ascii=False),
                            }
                            for k, v in (fields or {}).items():
                                row[normalize_key(k)] = v

                            results.append(row)
                            save_master_and_per_folder(results)
                            seen.add(rel); new_found += 1
                            move_to_done_local(full)

                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            results.append({
                                "folder": os.path.dirname(rel) or "root",
                                "relpath": rel,
                                "file": name,
                                "method": "error",
                                "error": str(e),
                                "raw_text": "",
                            })
                            save_master_and_per_folder(results)
                            seen.add(rel)
                            print(f"⚠️ 파일 처리 오류({rel}): {e}")

            if new_found == 0:
                print("…대기 중 (새 이미지 없음)")
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("🛑 사용자 중단.")
            break
        except Exception as e:
            print("⚠️ 루프 오류:", e)
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
