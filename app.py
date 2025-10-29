# app.py — Streamlit Web UI for OCR Agent

import os, io, json
import streamlit as st
import pandas as pd
import cv2
from dotenv import load_dotenv

# ====== local_ocr_agent 모듈 ======
from local_ocr_agent import (
    extract_by_template_dual,
    analyze_with_openai,
    find_container_by_full_ocr,
    autorotate_best,
    resize_long_side,
    normalize_key,
    USE_TEMPLATE,
    MAX_LONG_SIDE,
)

# ====== OCR 엔진 캐시 ======
@st.cache_resource(show_spinner=False)
def get_ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()
ocr_engine = get_ocr_engine()

# ====== Dropbox 헬퍼 ======
def list_dropbox_images(dbx, folder):
    IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff")
    paths = []
    try:
        res = dbx.files_list_folder(folder, recursive=True)
        while True:
            for e in res.entries:
                # FileMetadata만 (간단 판별)
                if hasattr(e, "path_lower") and hasattr(e, "client_modified"):
                    if e.path_lower.endswith(IMG_EXTS):
                        paths.append(e.path_lower)
            if not res.has_more:
                break
            res = dbx.files_list_folder_continue(res.cursor)
    except Exception as e:
        st.error(f"Dropbox list error: {e}")
    return paths

def download_dropbox_file(dbx, dbx_path):
    try:
        md, resp = dbx.files_download(dbx_path)
        return resp.content
    except Exception as e:
        st.error(f"Dropbox download error: {e}")
        return b""

# ====== 공통 처리 ======
def analyze_one_image_bytes(image_bytes: bytes, filename: str, use_template=True):
    """바이트 이미지 1장 처리 → row dict"""
    row = {"file": filename, "method": "", "type": "", "error": "", "raw_text": ""}
    # 임시 파일로 저장 후 OpenCV 로드
    tmp_path = os.path.join("tmp_uploads", filename)
    os.makedirs("tmp_uploads", exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)

    raw0 = cv2.imread(tmp_path)
    if raw0 is None:
        row["error"] = "imread_failed"
        return row
    raw0 = resize_long_side(raw0, MAX_LONG_SIDE)
    raw0 = autorotate_best(raw0)

    cont = find_container_by_full_ocr(raw0)
    if cont:
        row.update({"method": "fullocr", "type": "container", "container_number": cont})
        row["raw_text"] = json.dumps({"container_number": cont}, ensure_ascii=False)
        return row

    # 템플릿 → 부족하면 OpenAI 폴백
    fields, method, err, raw_text = {}, "template" if use_template and USE_TEMPLATE else "openai", "", ""
    if method == "template":
        fields, err = extract_by_template_dual(tmp_path, filename)
        key_fields = ["heat_no","bundle_no","size","grade","length","weight"]
        non_empty = sum(1 for k in key_fields if str(fields.get(k,"")).strip())
        if not fields or non_empty <= 1:
            method = "openai"

    if method == "openai":
        fields, raw_text, err2 = analyze_with_openai(tmp_path)
        err = f"{err};{err2}".strip(";")

    for k, v in (fields or {}).items():
        row[normalize_key(k)] = v
    row.update({"method": method, "error": err, "raw_text": raw_text or json.dumps(fields, ensure_ascii=False)})
    return row

def save_uploaded_file(uploaded_file, tmp_dir="tmp_uploads"):
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# ====== UI ======
st.set_page_config(page_title="OCR Agent Web", page_icon="📦", layout="wide")
st.title("📦 OCR Agent — Streamlit Web App")

load_dotenv("config.env", override=False)

st.sidebar.header("⚙️ 옵션")
api_key_in     = st.sidebar.text_input("OpenAI API Key (optional)", type="password", value=os.getenv("OPENAI_API_KEY",""))
use_template   = st.sidebar.checkbox("Use Label Templates (if available)", value=True)
show_preview   = st.sidebar.checkbox("Show image preview", value=True)
download_excel = st.sidebar.checkbox("Enable Excel download", value=True)

# Dropbox 옵션
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Dropbox 옵션")
use_dropbox_ui = st.sidebar.checkbox("Dropbox에서 바로 불러오기", value=False)
dbx_token      = st.sidebar.text_input("DROPBOX_TOKEN", type="password", value=os.getenv("DROPBOX_TOKEN",""))
dbx_folder     = st.sidebar.text_input("DROPBOX_FOLDER", value=os.getenv("DROPBOX_FOLDER","/work/LAVA의 팀 폴더/라바상사/CON_PHOTO"))

if api_key_in:
    os.environ["OPENAI_API_KEY"] = api_key_in  # local_ocr_agent가 참조

# 업로드 위젯
uploaded_files = st.file_uploader("📸 이미지 파일을 선택 또는 드래그하세요 (여러 장 가능)",
    type=["jpg","jpeg","png","bmp","webp","tif","tiff"], accept_multiple_files=True)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    run_upload = st.button("▶️ 업로드 분석 시작", use_container_width=True)
with col2:
    run_dbx = st.button("▶️ Dropbox에서 가져와 분석", use_container_width=True)
with col3:
    clear_btn = st.button("🧹 화면 초기화", use_container_width=True)

if clear_btn:
    st.experimental_rerun()

rows = []

# 업로드 분석
if run_upload and uploaded_files:
    prog = st.progress(0, text="처리 중...")
    for i, uf in enumerate(uploaded_files, start=1):
        path = save_uploaded_file(uf)
        if show_preview:
            st.image(path, caption=uf.name, width=360)
        with open(path, "rb") as f:
            img_bytes = f.read()
        try:
            row = analyze_one_image_bytes(img_bytes, uf.name, use_template=use_template)
        except Exception as e:
            row = {"file": uf.name, "method": "", "type": "", "error": str(e), "raw_text": ""}
        rows.append(row)
        prog.progress(i/len(uploaded_files), text=f"처리 {i}/{len(uploaded_files)}")
    prog.empty()

# Dropbox 분석
if run_dbx:
    if not (use_dropbox_ui and dbx_token and dbx_folder):
        st.error("Dropbox 사용 체크 + 토큰 + 폴더를 모두 입력해주세요.")
    else:
        try:
            import dropbox
            dbx = dropbox.Dropbox(dbx_token)
            st.info(f"폴더 스캔 중: {dbx_folder}")
            paths = list_dropbox_images(dbx, dbx_folder)
            if not paths:
                st.warning("이미지 파일을 찾지 못했습니다.")
            else:
                prog = st.progress(0, text="Dropbox에서 다운로드/분석 중...")
                for i, p in enumerate(paths, start=1):
                    content = download_dropbox_file(dbx, p)
                    if not content:
                        continue
                    fname = os.path.basename(p)
                    if show_preview:
                        st.image(content, caption=fname, width=360)
                    try:
                        row = analyze_one_image_bytes(content, fname, use_template=use_template)
                    except Exception as e:
                        row = {"file": fname, "method": "", "type": "", "error": str(e), "raw_text": ""}
                    rows.append(row)
                    prog.progress(i/len(paths), text=f"처리 {i}/{len(paths)}")
                prog.empty()
        except Exception as e:
            st.error(f"Dropbox 초기화 실패: {e}")

# 결과 표시/다운로드
if rows:
    df = pd.DataFrame(rows)
    preferred = [
        "file","method","type","container_number",
        "heat_no","bundle_no","size","grade","length","weight",
        "barcode","qr_numbers","error","raw_text",
    ]
    cols = list(df.columns)
    df = df[[c for c in preferred if c in cols] + [c for c in cols if c not in preferred]]

    st.subheader("📊 결과")
    st.dataframe(df, use_container_width=True, height=460)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ CSV 다운로드", data=csv, file_name="ocr_results.csv", mime="text/csv")
    if download_excel:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as xw:
            df.to_excel(xw, index=False, sheet_name="results")
        st.download_button(
            "⬇️ Excel 다운로드",
            data=bio.getvalue(),
            file_name="ocr_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("왼쪽 옵션을 설정하고, 이미지를 업로드하거나 Dropbox에서 불러온 뒤 분석 버튼을 눌러주세요.")
