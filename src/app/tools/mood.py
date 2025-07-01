# src/app/tools/mood.py
"""
mood
====

Tool for detecting user emotional state and tone preferences from free-text input.

Behavior
--------
- Automatically detects language (Spanish or English).
- Runs **TextBlob** sentiment analysis for English.
- Uses **pysentimiento / RoBERTuito** for Spanish.

Mapping rules convert raw sentiment into:
* mood: {'happy', 'sad', 'angry', 'neutral'}
* style: Spanish tone description for LangChain prompts
* emoji: Unicode emoji representing the mood

Returns
-------
dict
    Dictionary with fields:
    - 'mood'   : str
    - 'style'  : str
    - 'emoji'  : str
"""
from langdetect import detect           # idioma
from langchain_core.tools import tool
from textblob import TextBlob           # EN
from pysentimiento import create_analyzer  # ES

sa_es = create_analyzer(task="sentiment", lang="es")

_EMOJI = {"happy": "😀", "sad": "😢", "angry": "😡", "neutral": "🙂"}

@tool(description="Detects the emotional state (happy|sad|angry|neutral) "\
                  "and returns {'mood','style','emoji'}")
def detect_mood(text: str) -> dict:
    """
    Infer mood, tone style, and emoji from user input text.

    Parameters
    ----------
    text : str
        Raw user input in any language (English, Spanish).

    Returns
    -------
    dict
        Keys:
        - 'mood'   : str → e.g., 'happy', 'sad'
        - 'style'  : str → tone guide for downstream prompts
        - 'emoji'  : str → visual mood indicator
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
