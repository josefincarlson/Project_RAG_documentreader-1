# =========================
# chunking.py - Delar upp text i sökbara textstycken
# =========================
#
# Denna fil ansvarar för att normalisera text, dela upp dokument i mindre
# textstycken (chunks) och filtrera bort lågkvalitativa textstycken inför
# embeddings och sökning. Om du vill ändra hur text delas upp eller filtreras
# innan retrieval är detta rätt fil.


import re

from src.config import CHUNKING_SETTINGS

# =========================
# Textnormalisering - förbereder text för chunking
# =========================
#
# Denna del innehåller hjälpfunktioner för att normalisera råtext innan
# chunking. Här standardiseras radslut, styckegränser bevaras och onödiga
# mellanslag städas bort. Syftet är att ge mer stabil chunking och minska
# risken att radbrytningar eller formateringsbrus påverkar retrieval negativt.

def normalize_text(text: str) -> str:
    """
    Normaliserar text genom att städa radbrytningar och onödiga mellanslag.
    """
    if text is None:
        raise ValueError("text saknas.")

    if not isinstance(text, str):
        raise TypeError("text måste vara en sträng.")
    
    text = text.replace("\r\n", "\n").replace("\r", "\n")   # Gör radslut lika
    text = re.sub(r"\n\s*\n+", "\n\n", text)                # Bevara styckesgränser (två eller fler radbrytningar)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)            # Byt enkla radbrytningar inne i löpande text mot mellanslag
    text = re.sub(r"[ \t]+", " ", text)                     # Städa extra mellanslag

    return text.strip()


def estimate_token_count(text: str) -> int:
    """
    Grov token-estimering för lokal chunking utan extern tokenizer.

    Räknar ord och skiljetecken som separata token-liknande delar för att få
    stabilare chunkstorlek än ren teckenlängd, utan extra beroenden.
    """
    if text is None:
        raise ValueError("text saknas.")

    if not isinstance(text, str):
        raise TypeError("text måste vara en sträng.")

    if not text.strip():
        return 0

    parts = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return len(parts)


def measure_text_length(text: str, length_unit: str) -> int:
    """
    Mäter textlängd i vald längdenhet.
    """
    if text is None:
        raise ValueError("text saknas.")

    if not isinstance(text, str):
        raise TypeError("text måste vara en sträng.")

    if length_unit == "character":
        return len(text)

    if length_unit == "token_estimate":
        return estimate_token_count(text)

    raise ValueError(
        f"length_unit '{length_unit}' stöds inte. Tillåtna värden är 'character' och 'token_estimate'."
    )


def _join_chunk_parts(left: str, right: str) -> str:
    """
    Bygger ihop två chunkdelar med tydlig separator.
    """
    if not left:
        return right

    if not right:
        return left

    return f"{left}\n\n{right}"


def _split_sentence_if_needed(
    sentence: str,
    max_chunk_size: int,
    length_unit: str,
) -> list[str]:
    """
    Delar upp en för lång mening i mindre delar baserat på ord.
    """
    words = sentence.split()

    if not words:
        return []

    parts = []
    current_part = ""

    for word in words:
        candidate = f"{current_part} {word}".strip()

        if measure_text_length(candidate, length_unit) <= max_chunk_size:
            current_part = candidate
        else:
            if current_part:
                parts.append(current_part)
                current_part = word
            else:
                # Fallback om ett enskilt "ord" är extremt långt.
                parts.append(word)
                current_part = ""

    if current_part:
        parts.append(current_part)

    return parts


def split_paragraph_to_units(
    paragraph: str,
    max_chunk_size: int,
    length_unit: str,
) -> list[str]:
    """
    Delar upp ett stycke i mindre enheter som var för sig håller maxstorleken.

    Strategi:
    1. Behåll hela stycket om det redan får plats.
    2. Försök dela på meningar.
    3. Dela för långa meningar på ord.
    """
    if not paragraph:
        return []

    if measure_text_length(paragraph, length_unit) <= max_chunk_size:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if not sentences:
        return _split_sentence_if_needed(paragraph, max_chunk_size, length_unit)

    units = []
    current_unit = ""

    for sentence in sentences:
        sentence_parts = [sentence]

        if measure_text_length(sentence, length_unit) > max_chunk_size:
            sentence_parts = _split_sentence_if_needed(
                sentence,
                max_chunk_size=max_chunk_size,
                length_unit=length_unit,
            )

        for sentence_part in sentence_parts:
            candidate = f"{current_unit} {sentence_part}".strip()

            if measure_text_length(candidate, length_unit) <= max_chunk_size:
                current_unit = candidate
            else:
                if current_unit:
                    units.append(current_unit)
                current_unit = sentence_part

    if current_unit:
        units.append(current_unit)

    return units


def build_overlap_tail(text: str, overlap_size: int, length_unit: str) -> str:
    """
    Returnerar slutet av en chunk som overlap till nästa chunk.
    """
    if overlap_size <= 0 or not text.strip():
        return ""

    words = text.split()

    if not words:
        return ""

    tail_words = []

    for word in reversed(words):
        candidate_words = [word] + tail_words
        candidate = " ".join(candidate_words)

        if measure_text_length(candidate, length_unit) > overlap_size:
            break

        tail_words = candidate_words

    return " ".join(tail_words)


# =========================
# Chunkfilter - tar bort brus innan embeddings
# =========================
#
# Denna del innehåller hjälpfunktioner som filtrerar bort textstycken med
# lågt informationsvärde, till exempel metadata, dokumentstyrning och andra
# återkommande PDF-delar som riskerar att störa sökning och retrieval.

def get_low_value_min_length_for_extension(file_extension: str | None) -> int:
    """
    Returnerar lämplig min_length för lågkvalitetsfilter beroende på filtyp.
    Om filtypen saknas eller inte finns konfigurerad används standardvärdet.
    """
    extension_settings = CHUNKING_SETTINGS.get(
        "low_value_filter_min_length_per_extension",
        {}
    )
    default_min_length = CHUNKING_SETTINGS.get("min_chunk_size", 120)

    if file_extension is None:
        return default_min_length

    if not isinstance(file_extension, str):
        raise TypeError("file_extension måste vara en sträng eller None.")

    cleaned_extension = file_extension.strip().lower()

    if not cleaned_extension:
        return default_min_length

    return extension_settings.get(cleaned_extension, default_min_length)


def is_low_value_chunk(text: str, min_length: int = 120) -> bool:
    """
    Avgör om en chunk har lågt informationsvärde och bör filtreras bort
    innan embeddings skapas.

    Funktionen försöker fånga chunks som mest består av metadata,
    navigationshjälp, dokumentstyrning eller annat brus som ofta försämrar
    retrieval-kvaliteten, särskilt i PDF-dokument.

    Args:
        text: Chunkens textinnehåll.
        min_length: Minsta längd för att chunken ska anses ha rimligt innehållsvärde.

    Returns:
        True om chunken bör filtreras bort, annars False.
    """
    if text is None:
        raise ValueError("text saknas.")

    if not isinstance(text, str):
        raise TypeError("text måste vara en sträng.")

    if not isinstance(min_length, int):
        raise TypeError("min_length måste vara ett heltal.")

    if min_length < 0:
        raise ValueError("min_length kan inte vara negativ.")

    cleaned = text.strip().lower()

    if not cleaned:
        return True

    if len(cleaned) < min_length:
        return True

    noisy_patterns = [
        "sokord for dokumentassistent",
        "sökord för dokumentassistent",
        "nar anvands detta dokument",
        "när används detta dokument",
        "relaterade dokument",
        "exempel pa fragor",
        "exempel på frågor",
        "dokumentagare",
        "dokumentägare",
        "gallande dokument",
        "gällande dokument",
        "dokumentregister",
        "styrande dokument",
        "dokumentvard och revidering",
        "dokumentvård och revidering",
        "fraggebank for dokumentassistent",
        "frågebank för dokumentassistent",
        "forvantat dokument",
        "förväntat dokument",
        "nyckelord som bor matcha",
        "nyckelord som bör matcha",
        "fiktivt exempel for rag-testning",
        "fiktivt exempel för rag-testning",
        "demo- och testmaterial for dokumentassistent/rag",
        "demo- och testmaterial för dokumentassistent/rag",
    ]

    noisy_match_count = sum(1 for pattern in noisy_patterns if pattern in cleaned)
    noisy_length_limit = max(min_length * 4, 600)

    if noisy_match_count >= 2 and len(cleaned) <= noisy_length_limit:
        return True

    if (
        "vanliga fragor" in cleaned or "vanliga frågor" in cleaned
    ) and len(cleaned) <= noisy_length_limit:
        return True

    return False


def filter_low_value_chunks(chunks: list[str], min_length: int = 120) -> list[str]:
    """
    Filtrerar bort chunks med lågt informationsvärde.

    Denna funktion används efter chunking och före embeddings för att minska
    mängden brus i retrieval, till exempel sökordslistor, dokumentstyrning,
    sidnära metadata och andra chunks som inte hjälper modellen att besvara frågor.

    Args:
        chunks: Lista med textchunks.
        min_length: Minsta längd för att en chunk ska behållas om den inte matchar
            något känt brusmönster.

    Returns:
        En lista med filtrerade chunks.

    Raises:
        ValueError: Om chunks saknas eller om alla chunks filtreras bort.
        TypeError: Om chunks inte är en lista eller om min_length har fel typ.
    """
    if chunks is None:
        raise ValueError("chunks saknas.")

    if not isinstance(chunks, list):
        raise TypeError("chunks måste vara en lista.")

    if not isinstance(min_length, int):
        raise TypeError("min_length måste vara ett heltal.")

    if min_length < 0:
        raise ValueError("min_length kan inte vara negativ.")

    filtered_chunks = [
        chunk for chunk in chunks
        if not is_low_value_chunk(chunk, min_length=min_length)
    ]

    if not filtered_chunks:
        raise ValueError(
            "Alla chunks filtrerades bort. Kontrollera dokumentets innehåll eller filterreglerna."
        )

    return filtered_chunks


# =========================
# Styckesbaserad chunking
# =========================
#
# Denna del innehåller logik för att dela upp text i sökbara chunks baserat
# på styckesgränser. Funktionen försöker bevara naturliga textblock genom att
# bygga chunks av hela stycken upp till en maximal längd. Korta chunks kan
# därefter slås ihop för att minska brus och ge mer användbara embeddings.

def chunk_text_by_paragraphs(
    text: str,
    max_chunk_size: int | None = None,
    min_chunk_size: int | None = None,
    length_unit: str | None = None,
    overlap_size: int | None = None,
) -> list[str]:
    """
    Delar upp text i paragrafmedvetna chunks med token-estimat och overlap.
    """
    if max_chunk_size is None:
        max_chunk_size = CHUNKING_SETTINGS["max_chunk_size"]

    if min_chunk_size is None:
        min_chunk_size = CHUNKING_SETTINGS["min_chunk_size"]

    if length_unit is None:
        length_unit = CHUNKING_SETTINGS["length_unit"]

    if overlap_size is None:
        overlap_size = CHUNKING_SETTINGS.get("overlap_size", 0)

    if not isinstance(max_chunk_size, int):
        raise TypeError("max_chunk_size måste vara ett heltal.")

    if not isinstance(min_chunk_size, int):
        raise TypeError("min_chunk_size måste vara ett heltal.")

    if not isinstance(length_unit, str):
        raise TypeError("length_unit måste vara en sträng.")

    if not isinstance(overlap_size, int):
        raise TypeError("overlap_size måste vara ett heltal.")

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size måste vara större än 0.")

    if min_chunk_size < 0:
        raise ValueError("min_chunk_size kan inte vara negativ.")

    if min_chunk_size > max_chunk_size:
        raise ValueError("min_chunk_size kan inte vara större än max_chunk_size.")

    if overlap_size < 0:
        raise ValueError("overlap_size kan inte vara negativ.")

    if overlap_size >= max_chunk_size:
        raise ValueError("overlap_size måste vara mindre än max_chunk_size.")

    text = normalize_text(text)

    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units = []

    for paragraph in paragraphs:
        units.extend(
            split_paragraph_to_units(
                paragraph,
                max_chunk_size=max_chunk_size,
                length_unit=length_unit,
            )
        )

    chunks = []
    current_chunk = ""

    for unit in units:
        candidate_chunk = _join_chunk_parts(current_chunk, unit)

        if measure_text_length(candidate_chunk, length_unit) <= max_chunk_size:
            current_chunk = candidate_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)

                overlap_tail = build_overlap_tail(
                    current_chunk,
                    overlap_size=overlap_size,
                    length_unit=length_unit,
                )
                overlapped_chunk = _join_chunk_parts(overlap_tail, unit)

                if (
                    overlap_tail
                    and measure_text_length(overlapped_chunk, length_unit) <= max_chunk_size
                ):
                    current_chunk = overlapped_chunk
                else:
                    current_chunk = unit
            else:
                current_chunk = unit

    if current_chunk:
        chunks.append(current_chunk)

    if min_chunk_size > 0 and len(chunks) > 1:
        merged_chunks = []

        for chunk in chunks:
            if (
                merged_chunks
                and measure_text_length(chunk, length_unit) < min_chunk_size
                and measure_text_length(
                    _join_chunk_parts(merged_chunks[-1], chunk),
                    length_unit,
                ) <= max_chunk_size
            ):
                merged_chunks[-1] = _join_chunk_parts(merged_chunks[-1], chunk)
            else:
                merged_chunks.append(chunk)

        chunks = merged_chunks

    return chunks
