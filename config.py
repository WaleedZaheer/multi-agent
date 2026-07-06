"""
config.py
---------
Loads environment variables and creates the LLM client.
Keeping this in its own file means every agent asks for an LLM the same way,
and if you ever swap providers (Groq -> OpenAI -> Anthropic) you change ONE place.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # reads .env if present, same pattern as your RAG project


def get_llm(temperature: float = 0.3):
    """
    Returns a configured ChatGroq client.

    temperature controls randomness:
      - low (0.1-0.3)  -> consistent, factual, good for reviewing/deciding
      - higher (0.5-0.7) -> more creative, good for writing prose
    """
    api_key = os.getenv("GROQ_API_KEY")

    # Same dual-fallback pattern as your RAG project: env var first, then
    # Streamlit secrets (only relevant when running app.py, not main.py).
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Copy .env.example to .env and add your key, "
            "or export GROQ_API_KEY in your shell before running this project."
        )

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=api_key,
    )
