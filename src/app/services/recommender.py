# --- new imports ---
from dataclasses import dataclass, field
from typing import Tuple
from collections import defaultdict
import numpy as np, math, textwrap
from pathlib import Path
import json, uuid, time 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

@dataclass
class UserProfile:
    docs: set[str]                 = field(default_factory=set)
    qvecs: list[np.ndarray]        = field(default_factory=list)
    
class RecommendationService:
    def __init__(self,
                 vectordb,
                 persist_path: str = ".profiles.json",
                 flush_every: int = 10):
        self.vectordb   = vectordb
        self.emb        = OpenAIEmbeddings()         
        self.persist    = Path(persist_path)
        self.flush_every= flush_every
        self._writes    = 0                           # counter

        self._profiles: dict[str, UserProfile] = self._load_profiles()
        self._user_mood = defaultdict(lambda: {"mood":"neutral",
                                       "style":"profesional",
                                       "emoji":"🙂"})

    def log_sources(self, uid: str, sources: list[str]):
        self._profiles[uid].docs.update(sources)
        self._maybe_flush()

    def log_query(self, uid: str, query: str):
        vec = np.array(self.emb.embed_query(query), dtype=np.float32)
        self._profiles[uid].qvecs.append(vec)
        self._maybe_flush()

    def recommend(self, uid: str, k: int = 3, lambda_: float = 0.5):
        profile = self._profiles[uid]
        seen    = profile.docs

        ids, emb, meta, txt = self._get_vectors()
        unseen_idx = [i for i, m in enumerate(meta) if m["source"] not in seen]

        # --- COLD START --------------------------------------------------
        if not profile.qvecs:                         # no history yet
            picks = np.random.choice(unseen_idx, k, replace=False)
            return [self._build_payload(i, meta, txt, emb[picks].mean(0), emb)
                    for i in picks]

        # ----------------------------------------------------------------
        centroid = self._centroid(emb, meta, profile)
        ranked = self._mmr(centroid, emb, meta, unseen_idx, k, lambda_)
        return   [self._build_payload(i, meta, txt, centroid, emb) for i in ranked]


    # ---------- helpers ----------------------------------------------------
    def _get_vectors(self):
        data = self.vectordb._collection.get(
            include=["embeddings", "metadatas", "documents"])
        return data["ids"], np.array(data["embeddings"]), \
               data["metadatas"], data["documents"]

    @staticmethod
    def _cos(a, b):        # cosine similarity
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _centroid(self, emb, meta, profile: UserProfile):
        doc_vecs = emb[[i for i,m in enumerate(meta) if m["source"] in profile.docs]]

        q_vecs  = np.vstack(profile.qvecs) if profile.qvecs else np.empty((0, emb.shape[1]))

        if doc_vecs.size and q_vecs.size:
            return np.vstack([doc_vecs, q_vecs]).mean(axis=0)
        elif doc_vecs.size:
            return doc_vecs.mean(axis=0)
        else:
            return q_vecs.mean(axis=0)      


    def _mmr(self, query_vec, emb, meta, candidates, k, λ):
        selected = []
        while candidates and len(selected) < k:
            mmr_score, pick = -1, None
            for idx in candidates:
                rel = self._cos(query_vec, emb[idx])
                div = max(self._cos(emb[idx], emb[s]) for s in selected) if selected else 0
                if selected and meta[idx]["topic"] == meta[selected[-1]]["topic"]:
                    div += 0.15
                score = λ * rel - (1 - λ) * div
                if score > mmr_score:
                    mmr_score, pick = score, idx
            selected.append(pick)
            candidates.remove(pick)
        return selected
    
    def _build_payload(self, idx, meta, txt, centroid, emb) -> dict:
        title = Path(meta[idx]["source"]).stem.replace("_", " ")
        rel   = self._cos(centroid, emb[idx])
        lang  = "es" if meta[idx].get("lang") == "es" else "en"
        why   = (
            (f"Aún no has leído **{title}**; está relacionado "
            f"con tus intereses (similitud {rel:.2f}).")
            if lang == "es" else
            (f"We haven't shown you anything on **{title}** yet – "
            f"but it's close to what you asked ({rel:.2f}).")
        )
        return {
            "title": title.title(),
            "snippet": textwrap.shorten(txt[idx], 140),
            "why": why
        }
    
    def _maybe_flush(self):
        self._writes += 1
        if self._writes % self.flush_every == 0:
            self._save_profiles()

    def _save_profiles(self):
        data = {
            uid: {
                "docs":  list(p.docs),
                "qvecs": [v.tolist() for v in p.qvecs],
            } for uid, p in self._profiles.items()
        }
        tmp = self.persist.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.persist)    # atomic swap

    def _load_profiles(self) -> dict[str, UserProfile]:
        if not self.persist.exists():
            return defaultdict(UserProfile)
        raw = json.loads(self.persist.read_text())
        profs = defaultdict(UserProfile)
        for uid, p in raw.items():
            profs[uid].docs  = set(p["docs"])
            profs[uid].qvecs = [np.array(v, dtype=np.float32) for v in p["qvecs"]]
        return profs