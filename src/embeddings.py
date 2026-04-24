# =========================
# Embeddings för text och frågor
# =========================
#
# Laddar embedding-modellen och omvandlar dokumenttext och frågor till vektorer.
# Embedding-modellen styrs via config och inte från UI:t.


import time

import streamlit as st

from src.config_utils import get_active_embedding_config, validate_embedding_policy


# =========================
# Modelladdning och cache
# =========================
#
# Cachear embedding-modellen som Streamlit-resurs för att undvika onödiga omladdningar.

@st.cache_resource
def _load_sentence_transformer(model_name: str):
    """
    Laddar embedding-modellen en gång och cachear den som en Streamlit-resurs.
    Detta gör att modellen inte laddas om vid varje rerun.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# =========================
# Aktiv embedding-modell
# =========================
#
# Läser ut den aktiva embedding-modellen och validerar att den får användas.

def get_embedding_model():
    """
    Returnerar aktiv embedding-modell.

    Validerar först att embedding-policyn tillåter aktuell modell och använder
    därefter Streamlits resurscache för att undvika onödiga omladdningar.
    """
    config = validate_embedding_policy()
    provider = config["provider"]
    model_name = config["model_name"]

    if provider != "sentence_transformers":
        raise ValueError(
            f"Embedding-provider '{provider}' stöds inte ännu i embeddings.py."
        )

    return _load_sentence_transformer(model_name)


# =========================
# Embeddings för dokumenttext
# =========================
#
# Denna del innehåller logik för att skapa embeddings för dokumenttext och
# chunks. Här förbereds texten med eventuell dokumentprefix från config innan
# vektorer genereras med aktiv embedding-modell.

def embed_texts(texts: list[str]):
    """
    Genererar embeddings för en lista av dokument/chunks.
    """
    if not texts:
        raise ValueError("Listan med texter för embedding är tom.")

    if not all(isinstance(text, str) for text in texts):
        raise TypeError("Alla texter i 'texts' måste vara strängar.")
    
    config = validate_embedding_policy()
    model = get_embedding_model()

    prepared_texts = [
        f"{config['document_prefix']}{text}"
        for text in texts
    ]

    return model.encode(
        prepared_texts,
        normalize_embeddings=config["normalize_embeddings"],
        show_progress_bar=False,
    )


# =========================
# Embedding för användarfrågor
# =========================
#
# Denna del innehåller logik för att skapa embedding för användarens fråga.
# Frågan valideras och förbereds med eventuell query-prefix från config innan
# vektorn genereras med aktiv embedding-modell.

def embed_query(query: str):
    """
    Genererar embedding för användarens fråga.
    """
    if not query or not query.strip():
        raise ValueError("Frågan som ska embedda är tom.")

    query = query.strip()
    
    config = validate_embedding_policy()
    model = get_embedding_model()

    prepared_query = f"{config['query_prefix']}{query}"

    return model.encode(
        prepared_query,
        normalize_embeddings=config["normalize_embeddings"],
        show_progress_bar=False,
    )


# =========================
# Metadata om aktiv embedding-modell
# =========================
#
# Denna del returnerar metadata om den aktiva embedding-konfigurationen.
# Informationen kan användas för loggning, debugging och för att spara
# spårbar information tillsammans med index eller sökresultat.

def get_embedding_metadata() -> dict:
    """
    Returnerar metadata om aktiv embedding-modell.
    Kan användas för loggning, debugging eller index-metadata.
    """
    config = get_active_embedding_config()

    return {
        "embedding_model_key": config["model_key"],
        "embedding_provider": config["provider"],
        "embedding_model_name": config["model_name"],
        "embedding_execution_mode": config["execution_mode"],
        "normalize_embeddings": config["normalize_embeddings"],
        "query_prefix": config["query_prefix"],
        "document_prefix": config["document_prefix"],
    }
