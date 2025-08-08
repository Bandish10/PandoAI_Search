import os
import json
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import Tuple

# ----------------------------
# --------- SETTINGS ---------
# ----------------------------
MODEL_NAME = "abhinand/MedEmbed-base-v0.1"  # HF model (auto-download)
EXCEL_PATH = "Manipal_with_abbreviations.xlsx"

# Keep separate caches per model
SAFE_MODEL_DIRNAME = MODEL_NAME.replace("/", "-")
ARTIFACTS_DIR = os.path.join("vector_store", SAFE_MODEL_DIRNAME)
EMB_DIR = os.path.join(ARTIFACTS_DIR, "embeddings")
INDEX_DIR = os.path.join(ARTIFACTS_DIR, "faiss_index")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")
EMB_NPY = os.path.join(EMB_DIR, "embeddings.npy")
ROWMAP_CSV = os.path.join(EMB_DIR, "rowmap.csv")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")

EMBED_BATCH_SIZE = 128

st.set_page_config(page_title="Medical Vector Search", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# --------- HELPERS ----------
# ----------------------------
def ensure_dirs():
    os.makedirs(EMB_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

def file_fingerprint(paths: list) -> str:
    """Hash mtime+size of files (used to invalidate cache if Excel changes)."""
    h = hashlib.sha256()
    for p in paths:
        try:
            stat = os.stat(p)
            h.update(p.encode())
            h.update(str(stat.st_mtime_ns).encode())
            h.update(str(stat.st_size).encode())
        except FileNotFoundError:
            h.update(p.encode())
            h.update(b"missing")
    return h.hexdigest()

def normalize_0_1(sim: np.ndarray) -> np.ndarray:
    # cosine in [-1, 1] -> [0, 1]
    return (sim + 1.0) / 2.0

def concat_text(desc: str, syn: str, abbr: str) -> str:
    parts = []
    if pd.notna(desc) and str(desc).strip(): parts.append(str(desc))
    if pd.notna(syn) and str(syn).strip(): parts.append(str(syn))
    if pd.notna(abbr) and str(abbr).strip(): parts.append(str(abbr))
    return " [SEP] ".join(parts)

@st.cache_resource(show_spinner=False)
def load_model(name: str) -> SentenceTransformer:
    # Auto-downloads from HF on first call; cached afterward
    return SentenceTransformer(name)

@st.cache_data(show_spinner=False)
def load_df(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    req = ["Billing Group", "Billing Subgroup", "Surgery Level", "Code", "Description", "Synonyms", "Abbreviations"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns in Excel: {miss}")
    return df

def build_or_load_store(model: SentenceTransformer, df: pd.DataFrame) -> Tuple[faiss.IndexFlatIP, np.ndarray, pd.Series]:
    """Return (faiss_index, embeddings, rowmap). Embeddings are L2-normalized."""
    ensure_dirs()

    # Fingerprint Excel + model name (model is remote; include name to separate caches)
    fp = file_fingerprint([EXCEL_PATH]) + "|" + MODEL_NAME

    # Try cache
    meta = {}
    if os.path.exists(META_PATH):
        try:
            meta = json.load(open(META_PATH, "r", encoding="utf-8"))
        except Exception:
            meta = {}

    cache_ok = (
        meta.get("fingerprint") == fp
        and os.path.exists(EMB_NPY)
        and os.path.exists(ROWMAP_CSV)
        and os.path.exists(FAISS_PATH)
    )

    if cache_ok:
        embs = np.load(EMB_NPY)
        rowmap = pd.read_csv(ROWMAP_CSV)["orig_idx"]
        index = faiss.read_index(FAISS_PATH)
        return index, embs, rowmap

    with st.spinner("Encoding embeddings (first run only)…"):
        texts = [
            concat_text(df.loc[i, "Description"], df.loc[i, "Synonyms"], df.loc[i, "Abbreviations"])
            for i in range(len(df))
        ]
        embs = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine == dot after normalization
        )

        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs.astype(np.float32))

        np.save(EMB_NPY, embs)
        pd.DataFrame({"orig_idx": np.arange(len(df))}).to_csv(ROWMAP_CSV, index=False)
        faiss.write_index(index, FAISS_PATH)

        meta = {
            "fingerprint": fp,
            "built_at": datetime.utcnow().isoformat() + "Z",
            "model_name": MODEL_NAME,
            "excel_path": EXCEL_PATH,
            "rows": len(df),
            "dim": int(dim),
            "index_type": "FlatIP-cosine",
        }
        json.dump(meta, open(META_PATH, "w", encoding="utf-8"), indent=2)

        return index, embs, pd.Series(np.arange(len(df)))

def faiss_search(index: faiss.Index, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]
    scores, idxs = index.search(query_vec.astype(np.float32), top_k)
    return scores[0], idxs[0]

def embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]

def build_results_df(df: pd.DataFrame, idxs: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    cols = ["Billing Group", "Billing Subgroup", "Surgery Level", "Code", "Description"]
    out = df.iloc[idxs].reset_index(drop=True)[cols]
    out["Score"] = normalize_0_1(scores).clip(0, 1)
    return out

# ----------------------------
# ---------- APP -------------
# ----------------------------
st.title("🩺 Medical Vector Search (FAISS + MedEmbed)")

# Load data + model
try:
    df = load_df(EXCEL_PATH)
except Exception as e:
    st.error(f"Failed to load Excel: {e}")
    st.stop()

try:
    model = load_model(MODEL_NAME)
except Exception as e:
    st.error(f"Failed to load model '{MODEL_NAME}': {e}")
    st.info("Tip: If you see an auth error, set a Hugging Face token via env var HF_HOME/HF_TOKEN or login with `huggingface-cli login`.")
    st.stop()

# Build/load store
index, embs, rowmap = build_or_load_store(model, df)

# Session state
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Controls (no live search; use a form so Enter triggers submit)
with st.form("search_form", clear_on_submit=False):
    query = st.text_input("Search", value=st.session_state.last_query, placeholder="Type your query and press Enter…")
    top_k = st.slider("Number of results", min_value=5, max_value=100, value=20, step=5)
    submitted = st.form_submit_button("Search")

if submitted:
    q = (query or "").strip()
    st.session_state.last_query = q
    if not q:
        st.warning("Please enter a query.")
        st.session_state.results = pd.DataFrame()
    else:
        try:
            qvec = embed_query(model, q)
            scores, idxs = faiss_search(index, qvec, top_k=top_k)
            st.session_state.results = build_results_df(df, idxs, scores)
        except Exception as e:
            st.error(f"Search failed: {e}")

# Metrics
m1, m2, m3 = st.columns([2,2,2])
m1.metric("Rows indexed", len(df))
m2.metric("Dim", int(embs.shape[1] if embs is not None else 0))
m3.metric("Index", "FlatIP (cosine)")

st.markdown("---")
st.subheader("Results")
if not st.session_state.results.empty:
    st.dataframe(st.session_state.results, use_container_width=True, hide_index=True)
else:
    st.caption("Enter a query and press **Search** (or hit Enter) to see results.")

st.markdown(
    """
    <hr style="border: 1px solid #1f2937; margin-top: 1rem; margin-bottom: 0.5rem;" />
    <div style="color:#9ca3af; font-size:0.9rem;">
      Using FAISS inner-product over L2-normalized embeddings (cosine). Scores normalized to 0–1.
    </div>
    """,
    unsafe_allow_html=True,
)
