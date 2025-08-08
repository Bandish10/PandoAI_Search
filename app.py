import os
import json
import re
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import Tuple, List
from huggingface_hub import snapshot_download

# =========================
#        SETTINGS
# =========================
MODEL_REPO = "abhinand/MedEmbed-base-v0.1"
EXCEL_PATH = "Manipal_with_abbreviations.xlsx"

# Model folder (separate from embeddings/index)
MODEL_CACHE_ROOT = "hf_models"
SAFE_MODEL_DIRNAME = MODEL_REPO.replace("/", "-")
MODEL_LOCAL_DIR = os.path.join(MODEL_CACHE_ROOT, SAFE_MODEL_DIRNAME)

# Artifacts (embeddings + FAISS + row meta)
ART_DIR = os.path.join("vector_store", SAFE_MODEL_DIRNAME)
EMB_DIR = os.path.join(ART_DIR, "embeddings")
IDX_DIR = os.path.join(ART_DIR, "faiss_index")
META_PATH = os.path.join(ART_DIR, "meta.json")
EMB_NPY = os.path.join(EMB_DIR, "embeddings.npy")
ROWMAP_CSV = os.path.join(EMB_DIR, "rowmap.csv")
ROWMETA_CSV = os.path.join(EMB_DIR, "rowmeta.csv")  # flags & combined text
FAISS_PATH = os.path.join(IDX_DIR, "faiss.index")

EMBED_BATCH_SIZE = 128
RERANK_MULTIPLIER = 10          # search k*RERANK_MULTIPLIER then re-rank
BOOST_MATCH = 0.12              # boost for matching intent
PENALIZE_CONTRADICT = 0.12      # penalty for contradicting intent

st.set_page_config(page_title="Medical Vector Search", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background-color: #0e1117; }
      .block-container { padding-top: 1.0rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
#        HELPERS
# =========================
def ensure_dirs():
    os.makedirs(MODEL_CACHE_ROOT, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)
    os.makedirs(IDX_DIR, exist_ok=True)

def list_all_files(root: str) -> List[str]:
    paths = []
    for base, _, files in os.walk(root):
        for f in files:
            paths.append(os.path.join(base, f))
    return paths

def file_fingerprint(paths: list) -> str:
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
    return (sim + 1.0) / 2.0

def concat_text(desc: str, syn: str, abbr: str) -> str:
    parts = []
    if pd.notna(desc) and str(desc).strip(): parts.append(str(desc))
    if pd.notna(syn) and str(syn).strip(): parts.append(str(syn))
    if pd.notna(abbr) and str(abbr).strip(): parts.append(str(abbr))
    return " [SEP] ".join(parts)

# intent patterns (simple, fast)
RE_UNILAT = re.compile(r"\b(unilateral|single|one|1\s*(?:side|sided)?)\b", re.I)
RE_BILAT  = re.compile(r"\b(bilateral|both|two|2\s*(?:side|sided)?)\b", re.I)

def extract_flags(text: str) -> Tuple[bool, bool]:
    t = text.lower()
    return (bool(RE_UNILAT.search(t)), bool(RE_BILAT.search(t)))

@st.cache_resource(show_spinner=False)
def ensure_model(repo_id: str, local_dir: str) -> str:
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return local_dir

@st.cache_resource(show_spinner=False)
def load_model(local_dir: str) -> SentenceTransformer:
    return SentenceTransformer(local_dir)

@st.cache_data(show_spinner=False)
def load_df(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    req = ["Billing Group", "Billing Subgroup", "Surgery Level", "Code", "Description", "Synonyms", "Abbreviations"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns in Excel: {miss}")
    return df

def build_or_load_store(model: SentenceTransformer, df: pd.DataFrame):
    """
    Returns: (faiss_index, rowmap: Series, rowmeta: DataFrame, dim: int)
    Embeddings are stored on disk; loaded with mmap for minimal RAM.
    """
    ensure_dirs()
    # include model files + excel in fingerprint
    model_files = list_all_files(MODEL_LOCAL_DIR)
    fp = file_fingerprint([EXCEL_PATH] + model_files)

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
        and os.path.exists(ROWMETA_CSV)
        and os.path.exists(FAISS_PATH)
    )

    if cache_ok:
        index = faiss.read_index(FAISS_PATH)
        embs = np.load(EMB_NPY, mmap_mode="r")  # mmap keeps memory low
        dim = int(embs.shape[1])
        del embs
        rowmap = pd.read_csv(ROWMAP_CSV)["orig_idx"]
        rowmeta = pd.read_csv(ROWMETA_CSV)
        return index, rowmap, rowmeta, dim

    with st.spinner("Encoding embeddings (first run only)…"):
        combined = [
            concat_text(df.loc[i, "Description"], df.loc[i, "Synonyms"], df.loc[i, "Abbreviations"])
            for i in range(len(df))
        ]
        flags = [extract_flags(c) for c in combined]
        rowmeta = pd.DataFrame({
            "orig_idx": np.arange(len(df), dtype=np.int32),
            "combined_lower": [c.lower() for c in combined],
            "has_unilateral": [int(f[0]) for f in flags],
            "has_bilateral":  [int(f[1]) for f in flags],
        })

        embs = model.encode(
            combined,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine == dot
        )
        dim = int(embs.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embs.astype(np.float32))

        np.save(EMB_NPY, embs)
        pd.DataFrame({"orig_idx": np.arange(len(df), dtype=np.int32)}).to_csv(ROWMAP_CSV, index=False)
        rowmeta.to_csv(ROWMETA_CSV, index=False)
        faiss.write_index(index, FAISS_PATH)

        meta = {
            "fingerprint": fp,
            "built_at": datetime.utcnow().isoformat() + "Z",
            "model_repo": MODEL_REPO,
            "model_local_dir": MODEL_LOCAL_DIR,
            "excel_path": EXCEL_PATH,
            "rows": int(len(df)),
            "dim": dim,
            "index_type": "FlatIP-cosine",
        }
        json.dump(meta, open(META_PATH, "w", encoding="utf-8"), indent=2)

        del embs
        return index, pd.Series(np.arange(len(df))), rowmeta, dim

def faiss_search(index: faiss.Index, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]
    scores, idxs = index.search(query_vec.astype(np.float32), top_k)
    return scores[0], idxs[0]

def embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]

def rerank_with_intent(df: pd.DataFrame, rowmeta: pd.DataFrame, idxs: np.ndarray, scores: np.ndarray, query: str):
    q_uni = bool(RE_UNILAT.search(query))
    q_bi  = bool(RE_BILAT.search(query))

    adj_scores = scores.copy()
    meta_sel = rowmeta.iloc[idxs]
    if q_uni:
        adj_scores += BOOST_MATCH * meta_sel["has_unilateral"].to_numpy()
        adj_scores -= PENALIZE_CONTRADICT * meta_sel["has_bilateral"].to_numpy()
    if q_bi:
        adj_scores += BOOST_MATCH * meta_sel["has_bilateral"].to_numpy()
        adj_scores -= PENALIZE_CONTRADICT * meta_sel["has_unilateral"].to_numpy()

    order = np.argsort(-adj_scores, kind="stable")  # stable keeps initial order for ties
    return idxs[order], adj_scores[order]

def build_results_df(df: pd.DataFrame, idxs: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    cols = ["Billing Group", "Billing Subgroup", "Surgery Level", "Code", "Description"]
    out = df.iloc[idxs].reset_index(drop=True)[cols]
    out["Score"] = normalize_0_1(scores).clip(0, 1)
    return out

# =========================
#          APP
# =========================
st.title("🩺 Medical Vector Search (FAISS + MedEmbed)")

# Model (cached)
try:
    local_dir = ensure_model(MODEL_REPO, MODEL_LOCAL_DIR)
    model = load_model(local_dir)
except Exception as e:
    st.error(f"Failed to download/load model '{MODEL_REPO}': {e}")
    st.info("If you see an auth error or rate-limit, run `huggingface-cli login` or set HF_TOKEN.")
    st.stop()

# Data
try:
    df = load_df(EXCEL_PATH)
except Exception as e:
    st.error(f"Failed to load Excel: {e}")
    st.stop()

# Vector store
index, rowmap, rowmeta, dim = build_or_load_store(model, df)

# --- Session state init (DO NOT bind to same keys as widgets) ---
if "query" not in st.session_state:
    st.session_state["query"] = ""
if "committed_query" not in st.session_state:
    st.session_state["committed_query"] = ""
if "results" not in st.session_state:
    st.session_state["results"] = pd.DataFrame()
if "top_k" not in st.session_state:
    st.session_state["top_k"] = 20

# Controls (use different widget keys to avoid the mutation error)
with st.form("search_form", clear_on_submit=False):
    query_input = st.text_input(
        "Search",
        value=st.session_state["query"],
        key="query_input",
        placeholder="Type your query (e.g., 'TKR single', 'TKR unilateral') and press Enter…"
    )
    topk_value = st.slider(
        "Number of results",
        min_value=5, max_value=100,
        value=st.session_state["top_k"],
        step=5, key="topk_slider"
    )
    submitted = st.form_submit_button("Search")

# Only when user submits do we compute/search and lock the query
if submitted:
    st.session_state["query"] = query_input
    st.session_state["top_k"] = int(topk_value)
    st.session_state["committed_query"] = (st.session_state["query"] or "").strip()

    if not st.session_state["committed_query"]:
        st.warning("Please enter a query.")
        st.session_state["results"] = pd.DataFrame()
    else:
        try:
            qvec = embed_query(model, st.session_state["committed_query"])
            # fetch extra for re-ranking
            k_big = min(st.session_state["top_k"] * RERANK_MULTIPLIER, index.ntotal)
            base_scores, base_idxs = faiss_search(index, qvec, top_k=k_big)
            # keyword-aware rerank
            rerank_idxs, rerank_scores = rerank_with_intent(
                df, rowmeta, base_idxs, base_scores, st.session_state["committed_query"]
            )
            # trim to top_k
            rerank_idxs = rerank_idxs[:st.session_state["top_k"]]
            rerank_scores = rerank_scores[:st.session_state["top_k"]]
            st.session_state["results"] = build_results_df(df, rerank_idxs, rerank_scores)
        except Exception as e:
            st.error(f"Search failed: {e}")

# Metrics (no need to keep embeddings in RAM)
m1, m2, m3 = st.columns([2,2,2])
m1.metric("Rows indexed", len(df))
m2.metric("Dim", int(dim))
m3.metric("Index", "FlatIP (cosine)")

st.markdown("---")
st.subheader("Results")
if not st.session_state["results"].empty:
    st.dataframe(st.session_state["results"], use_container_width=True, hide_index=True)
else:
    st.caption("Enter a query and press **Search** (or hit Enter) to see results.")

st.markdown(
    """
    <hr style="border: 1px solid #1f2937; margin-top: 1rem; margin-bottom: 0.5rem;" />
    <div style="color:#9ca3af; font-size:0.9rem;">
      FAISS inner-product over L2-normalized embeddings (cosine). Scores normalized to 0–1.
      Lightweight keyword-aware reranking fixes unilateral/bilateral edge cases.
    </div>
    """,
    unsafe_allow_html=True,
)
