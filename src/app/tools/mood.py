# src/app/tools/mood.py
from langdetect import detect           # idioma
from langchain_core.tools import tool
from textblob import TextBlob           # EN
from pysentimiento import create_analyzer  # ES

sa_es = create_analyzer(task="sentiment", lang="es")

_EMOJI = {"happy": "😀", "sad": "😢", "angry": "😡", "neutral": "🙂"}

@tool(description="Detecta el estado emocional (happy|sad|angry|neutral) "\
                  "y devuelve {'mood','style','emoji'}")
def detect_mood(text: str) -> dict:
    """
    Usa TextBlob si el texto está mayoritariamente en inglés;
    usa pysentimiento (RoBERTuito) si está en español.
    """
    try:
        lang = detect(text)             # 'en', 'es', …
    except Exception:
        lang = "en"

    if lang == "es":
        pred = sa_es.predict(text)
        pol  = pred.output              # POS / NEG / NEU
        mood = {"POS": "happy",
                "NEG": "angry",
                "NEU": "neutral"}[pol]
    else:                               # fallback inglés
        score = TextBlob(text).sentiment.polarity  # [-1,1] :contentReference[oaicite:3]{index=3}
        if score >  0.4: mood = "happy"
        elif score < -0.4: mood = "angry"
        elif score < -0.1: mood = "sad"
        else:           mood = "neutral"

    style = {
        "happy":  "alegre y motivador",
        "sad":    "cálido y empático",
        "angry":  "sereno y conciliador",
        "neutral":"profesional",
    }[mood]
    return {"mood": mood, "style": style, "emoji": _EMOJI[mood]}
