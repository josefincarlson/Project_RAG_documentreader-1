# =========================
# Svarsgenerering med språkmodeller
# =========================
#
# Validerar modellvalet och skickar anropet vidare till rätt provider.

import os
from functools import lru_cache
from typing import Any, Literal, Mapping

from dotenv import load_dotenv

from src.config_utils import get_llm_model_settings, validate_llm_policy

load_dotenv()

LLMProvider = Literal["ollama", "gemini"]


# =========================
# Huvudfunktion för LLM-anrop
# =========================
#
# Validerar indata och skickar anropet vidare till rätt providerfunktion.

def generate_response(
    provider: LLMProvider,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Genererar svar från vald LLM-provider och modell.
    """
    if not provider:
        raise ValueError("provider är tom.")

    if not model_name or not model_name.strip():
        raise ValueError("model_name är tom.")

    if not system_prompt or not system_prompt.strip():
        raise ValueError("system_prompt är tom.")

    if not user_prompt or not user_prompt.strip():
        raise ValueError("user_prompt är tom.")

    system_prompt = system_prompt.strip()
    user_prompt = user_prompt.strip()

    validate_llm_policy(provider=provider, model_name=model_name)
    settings = get_llm_model_settings(provider, model_name)

    if provider == "ollama":
        return _generate_ollama_response(
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            settings=settings,
        )

    if provider == "gemini":
        return _generate_gemini_response(
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            settings=settings,
        )

    raise ValueError(f"Okänd provider: {provider}")


# =========================
# Gemensamma hjälpfunktioner
# =========================
#
# Intern logik som delas av flera providers.

def _get_temperature(settings: Mapping[str, Any]) -> float:
    """
    Hämtar temperature-inställning från modellspecifika inställningar.
    """
    if "temperature" not in settings:
        raise ValueError("Modellinställningarna saknar 'temperature' i config.py.")

    temperature = settings["temperature"]

    if not isinstance(temperature, (int, float)):
        raise TypeError("'temperature' måste vara ett tal i config.py.")

    return float(temperature)


# =========================
# Ollama
# =========================
#
# Skickar chatmeddelanden till en lokal modell via Ollama.

def _generate_ollama_response(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    settings: Mapping[str, Any],
) -> str:
    """
    Genererar svar via Ollama.
    """
    import ollama

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": _get_temperature(settings),
        },
    )

    content = response.get("message", {}).get("content")
    if not content:
        raise ValueError("Ollama returnerade inget textsvar.")

    return content


# =========================
# Gemini-klient och Gemini-anrop
# =========================
#
# Skapar en cachad Gemini-klient och använder den för svarsgenerering.

@lru_cache(maxsize=1)
def _get_gemini_client():
    """
    Returnerar cachad Gemini-klient.

    Just nu används en enda klient per session eftersom appen endast
    arbetar med en aktiv API-nyckel. Om flera nycklar eller klient-
    konfigurationer ska stödjas senare bör cachelösningen utökas så
    att den cachear per konfiguration.
    """
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai-paketet är inte installerat. Kör: pip install google-genai"
        ) from exc

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError("GEMINI_API_KEY saknas i miljövariablerna.")

    return genai.Client(api_key=api_key)


def _generate_gemini_response(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    settings: Mapping[str, Any],
) -> str:
    """
    Genererar svar via Google Gemini.
    """
    try:
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "google-genai-paketet är inte korrekt installerat. Kör: pip install google-genai"
        ) from exc

    client = _get_gemini_client()

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=_get_temperature(settings),
    )

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=config,
    )

    if not getattr(response, "text", None):
        raise ValueError("Gemini returnerade inget textsvar.")

    return response.text.strip()
