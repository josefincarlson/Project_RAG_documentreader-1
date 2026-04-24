# =========================
# RAG-flödet från fråga till svar
# =========================
#
# Kopplar ihop retrieval, kontextbyggande och svarsgenerering.

import re

from src.config import RAG_SETTINGS
from src.llm import LLMProvider, generate_response
from src.retriever import semantic_search


# =========================
# Systemprompt för svarsgenerering
# =========================
#
# Styr att modellen ska svara kort, tydligt och bara utifrån kontexten.

SYSTEM_PROMPT = """
Du är en hjälpsam assistent som bara får svara utifrån den kontext du får.

Regler:
1. Svara endast utifrån kontexten.
2. Om svaret inte finns i kontexten, svara exakt: Det vet jag inte.
3. Du får inte gissa, anta eller spekulera.
4. Svara på svenska.
5. Svara kort, tydligt och direkt.
6. Lägg inte till information som inte stöds av kontexten.
7. Om användaren ber om en lista, svara med en kort lista.
8. Om användaren frågar om en process eller rutin, svara gärna i numrerade steg.
9. Om användaren frågar vad ett dokument handlar om, ge en kort sammanfattning utifrån kontexten.
10. Om användaren främst frågar var information finns, svara med dokumentnamn och relevant källplats i stället för att återge innehållet.
11. Avsluta alltid svaret med en kort källhänvisning.
12. Ange dokumentnamn och, om det finns i kontexten, även sida och/eller stycke.
13. Hitta inte på källor eller sidnummer. Använd bara källuppgifter som finns i kontexten.
""".strip()


# =========================
# Källinformation i text
# =========================
#
# Hjälpfunktioner för att läsa ut och strukturera källinformation, till exempel sidnummer.

def extract_page_from_text(text: str) -> int | None:
    """
    Försöker läsa ut sidnummer ur chunktexten, till exempel [Sida 35].
    """
    if not text:
        return None

    match = re.search(r"\[Sida\s+(\d+)\]", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


# =========================
# Kontext från sökresultat
# =========================
#
# Bygger ihop retrieval-resultat till en gemensam kontextsträng med källmetadata.

def build_context_from_results(
    results: list[dict],
    all_metadata: list[dict],
) -> str:
    """
    Bygger ihop hittade textstycken till en gemensam kontextsträng för LLM:en,
    inklusive källmetadata som dokumentnamn, sida och stycke.
    """
    if not results:
        return ""

    if all_metadata is None:
        raise ValueError("all_metadata saknas.")

    prompting_settings = RAG_SETTINGS.get("prompting", {})
    separator = prompting_settings.get("context_separator")

    if not separator:
        raise ValueError("RAG_SETTINGS saknar 'prompting.context_separator' i config.py.")

    context_parts = []

    for item in results:
        if "text" not in item:
            raise ValueError("Ett sökresultat saknar nyckeln 'text'.")

        if "index" not in item:
            raise ValueError("Ett sökresultat saknar nyckeln 'index'.")

        chunk_index = item["index"]

        if not isinstance(chunk_index, int):
            raise TypeError("Sökresultatets 'index' måste vara ett heltal.")

        if chunk_index < 0 or chunk_index >= len(all_metadata):
            raise ValueError("Sökresultatets index ligger utanför metadata-listan.")

        metadata = all_metadata[chunk_index]
        doc_name = metadata.get("doc", "Okänt dokument")
        chunk_number = metadata.get("chunk_index")
        category = metadata.get("category", "Okänd kategori")
        page_number = extract_page_from_text(item["text"])

        source_lines = [
            "[Källa]",
            f"Dokument: {doc_name}",
            f"Kategori: {category}",
        ]

        if isinstance(chunk_number, int):
            source_lines.append(f"Stycke: {chunk_number + 1}")

        if page_number is not None:
            source_lines.append(f"Sida: {page_number}")

        source_lines.append("Text:")
        source_lines.append(item["text"])

        context_parts.append("\n".join(source_lines))

    return separator.join(context_parts)


# =========================
# Användarprompt
# =========================
#
# Skapar den prompt som skickas till LLM:en utifrån fråga och kontext.

def generate_user_prompt(query: str, context: str) -> str:
    """
    Bygger användarprompten som skickas till LLM:en.
    """
    return f"""Fråga: {query}

Kontext:
{context}""".strip()


# =========================
# Identifiera frågetyp
# =========================
#
# Tolkar om frågan främst gäller sammanfattning eller var information finns.

def is_summary_query(query: str) -> bool:
    """
    Avgör om frågan sannolikt efterfrågar en sammanfattning eller de viktigaste
    punkterna i ett dokument, snarare än ett direkt faktasvar.
    """
    if query is None:
        raise ValueError("query saknas.")

    if not isinstance(query, str):
        raise TypeError("query måste vara en sträng.")

    cleaned = query.strip().lower()

    summary_patterns = [
        "top 3 viktigaste i dokumentet",
        "top tre viktigaste i dokumentet",
        "vad handlar dokumentet om",
        "vad innehaller dokumentet",
        "vad innehåller dokumentet",
        "sammanfatta dokumentet",
        "sammanfatta hela dokumentet",
        "huvudpunkter i dokumentet",
        "huvudpunkterna i dokumentet",
    ]

    return any(pattern in cleaned for pattern in summary_patterns)


def is_location_query(query: str) -> bool:
    """
    Avgör om frågan efterfrågar var information finns, snarare än själva innehållet.
    """
    if query is None:
        raise ValueError("query saknas.")

    if not isinstance(query, str):
        raise TypeError("query måste vara en sträng.")

    cleaned = query.strip().lower()

    location_patterns = [
        "vart hittar jag",
        "var hittar jag",
        "var kan jag läsa",
        "vilket dokument",
        "i vilket dokument",
        "var står det",
        "var finns",
        "finns det något dokument om",
        "vilken policy",
        "vilket dokument handlar om",
        "vad heter dokumentet",
    ]

    return any(pattern in cleaned for pattern in location_patterns)


# =========================
# Huvudflöde för RAG-frågor
# =========================
#
# Kör hela flödet från validering och retrieval till färdigt svar.

def answer_query(
    query: str,
    chunks: list[str],
    chunk_embeddings,
    all_metadata: list[dict],
    embed_query_func,
    provider: LLMProvider,
    model_name: str,
    k: int | None = None,
    min_score: float | None = None,
) -> tuple[str, list[dict], str]:
    """
    Kör hela RAG-flödet: retrieval, kontextbyggande och svarsgenerering.
    """
    if not query or not query.strip():
        raise ValueError("query är tom.")
    
    query = query.strip()

    if not chunks:
        raise ValueError("chunks är tom.")

    if chunk_embeddings is None:
        raise ValueError("chunk_embeddings saknas.")

    if all_metadata is None:
        raise ValueError("all_metadata saknas.")

    if len(all_metadata) != len(chunks):
        raise ValueError(
            f"Antal metadata-poster ({len(all_metadata)}) matchar inte antal chunks ({len(chunks)})."
        )
    
    if embed_query_func is None:
        raise ValueError("embed_query_func saknas.")
    
    retrieval_settings = RAG_SETTINGS.get("retrieval", {})
    summary_settings = RAG_SETTINGS.get("summary_retrieval", {})

    if "top_k" not in retrieval_settings:
        raise ValueError("RAG_SETTINGS saknar 'retrieval.top_k' i config.py.")

    if "min_score" not in retrieval_settings:
        raise ValueError("RAG_SETTINGS saknar 'retrieval.min_score' i config.py.")

    if summary_settings.get("enabled", False):
        if "top_k" not in summary_settings:
            raise ValueError("RAG_SETTINGS saknar 'summary_retrieval.top_k' i config.py.")

        if "min_score" not in summary_settings:
            raise ValueError("RAG_SETTINGS saknar 'summary_retrieval.min_score' i config.py.")
    
    if k is None:
        k = retrieval_settings["top_k"]

    if min_score is None:
        min_score = retrieval_settings["min_score"]

    effective_k = k
    effective_min_score = min_score
    summary_query = is_summary_query(query)

    if summary_query and summary_settings.get("enabled", False):
        effective_k = max(k, summary_settings.get("top_k", k))
        effective_min_score = min(
            min_score,
            summary_settings.get("min_score", min_score),
        )


    results = semantic_search(
        query=query,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        embed_query_func=embed_query_func,
        k=effective_k,
        threshold=effective_min_score,
    )

    if not results:
        return "Det vet jag inte.", [], ""

    location_query = is_location_query(query)

    context = build_context_from_results(results, all_metadata)

    if location_query:
        user_prompt = f"""Fråga: {query}

    Kontext:
    {context}

    Instruktion:
    Användaren frågar efter var informationen finns. Svara endast med dokumentnamn, dokumentkod, titel eller tydlig källplats om den framgår i kontexten. Besvara inte frågan med själva innehållet i policyn eller dokumentet."""
    else:
        user_prompt = generate_user_prompt(query, context)

    answer = generate_response(
        provider=provider,
        model_name=model_name,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    return answer, results, context
