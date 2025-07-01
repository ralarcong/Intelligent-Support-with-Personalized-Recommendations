# src/app/services/recommender.py
"""
recommender
===========

Lightweight content-based recommender that operates on the same Chroma
collection as the RAG service.

For every user it stores:
* ``docs``   – set of file paths already shown.
* ``qvecs``  – stack of query embeddings (OpenAI).

At recommendation time the centroid of read-docs + query vectors is
computed and Maximum-Marginal-Relevance (MMR) is applied to select *k*
unseen documents while promoting topical diversity.
"""
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
    """
    Personalised document recommender.

    Parameters
    ----------
    vectordb : chromadb.api.models.Collection
        Chroma collection shared with the Q&A service.
    persist_path : str, default ``".profiles.json"``
        JSON file used to persist user profiles across restarts.
    flush_every : int, default 10
        Number of write operations after which the JSON is flushed.

    Attributes
    ----------
    emb : langchain_openai.OpenAIEmbeddings
        Embedding client to vectorise new queries on the fly.
    _profiles : dict[str, UserProfile]
        In-memory store of per-user document sets and query vectors.
    """
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
        """
        Mark the given source files as *read* by the user.

        Parameters
        ----------
        uid : str
            User identifier.
        sources : list[str]
            List of file paths retrieved by the RAG answer.
        """
        self._profiles[uid].docs.update(sources)
        self._maybe_flush()

    def log_query(self, uid: str, query: str):
        """
        Embed and store a new user query for later centroid calculation.

        Parameters
        ----------
        uid : str
            User identifier.
        query : str
            Raw query string (in any language supported by the embedding
            model).
        """
        vec = np.array(self.emb.embed_query(query), dtype=np.float32)
        self._profiles[uid].qvecs.append(vec)
        self._maybe_flush()

    def recommend(self, uid: str, k: int = 3, lambda_: float = 0.5):
        """
        Return *k* unseen, diversified recommendations.

        Parameters
        ----------
        uid : str
            Target user.
        k : int, default 3
            Number of documents to suggest.
        lambda_ : float, default 0.5
            Relevance/diversity trade-off for MMR (``1 = purely
            relevance``, ``0 = purely diversity``).

        Returns
        -------
        list of dict
            Each dictionary contains:
            ``title``   : str – human-friendly title  
            ``snippet`` : str – 140-char preview  
            ``why``     : str – explanation in EN/ES depending on doc

        Notes
        -----
        * Cold-start: if the user has no query vectors yet, *k* random
          unseen documents are returned.
        * Diversity boost: consecutive documents with identical
          ``topic`` receive an extra penalty.
        """
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
        """Fetch all document embeddings, metadata, and raw text from the vector store."""
        data = self.vectordb._collection.get(
            include=["embeddings", "metadatas", "documents"])
        return data["ids"], np.array(data["embeddings"]), \
               data["metadatas"], data["documents"]

    @staticmethod
    def _cos(a, b):        # cosine similarity
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _centroid(self, emb, meta, profile: UserProfile):
        """Calculate the user profile centroid from seen documents and past query vectors."""
        doc_vecs = emb[[i for i,m in enumerate(meta) if m["source"] in profile.docs]]

        q_vecs  = np.vstack(profile.qvecs) if profile.qvecs else np.empty((0, emb.shape[1]))

        if doc_vecs.size and q_vecs.size:
            return np.vstack([doc_vecs, q_vecs]).mean(axis=0)
        elif doc_vecs.size:
            return doc_vecs.mean(axis=0)
        else:
            return q_vecs.mean(axis=0)      


    def _mmr(self, query_vec, emb, meta, candidates, k, λ):
        """Select k diverse documents from candidates using MMR with topic penalization."""
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
        """Format a recommendation payload with title, snippet, and a relevance explanation."""
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
        """Flush user profiles to disk after every N writes (atomic swap for safety)."""
        self._writes += 1
        if self._writes % self.flush_every == 0:
            self._save_profiles()

    def _save_profiles(self):
        """Persist all user profiles to JSON file on disk (overwrites previous version)."""
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
        """Load user profiles from disk into memory, or initialize empty if none found."""
        if not self.persist.exists():
            return defaultdict(UserProfile)
        raw = json.loads(self.persist.read_text())
        profs = defaultdict(UserProfile)
        for uid, p in raw.items():
            profs[uid].docs  = set(p["docs"])
            profs[uid].qvecs = [np.array(v, dtype=np.float32) for v in p["qvecs"]]
        return profs