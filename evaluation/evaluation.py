# =========================
# evaluation.py - evaluering av RAG-pipeline
# =========================
#
# Denna modul kör testfall mot RAG-pipelinen och sparar resultat i CSV.
# Fokus ligger på att mäta:
# - retrieval-träff
# - svarskvalitet
# - svarstid
# - skillnader mellan olika LLM-modeller (intial tanke, men pga gratisversion av Gemini körs enbart ollama nu)

import csv
import re
import sys
import time
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import chunk_text_by_paragraphs, filter_low_value_chunks
from src.embeddings import embed_query, embed_texts
from src.loaders import load_document
from src.rag_pipeline import answer_query


MIN_CHUNKS_AFTER_FILTER = 3
MIN_FILTER_RETENTION_RATIO = 0.30


TEST_CASES = [
    {
        "query": "Hur gör jag för att ställa in min iPhone mot företaget?",
        "ideal_answer": (
            "Guide 5 beskriver att användaren ska installera Outlook, "
            "Authenticator och Company Portal, logga in med företagskontot och "
            "slutföra registreringen i Company Portal innan jobbmejl och policyer "
            "syns i mobilen."
        ),
        "answer_keywords": [
            "company portal", "outlook", "authenticator", "företagskonto",
            "registreringen", "jobbmejl",
        ],
        "retrieval_keywords": [
            "guide 5", "iphone", "company portal", "outlook",
            "authenticator", "jobbmobil",
        ],
        "accepted_doc_refs": ["guide 5", "iphone"],
        "soft_min_keyword_matches": 2,
    },
    {
        "query": "Hur gör jag om jag glömt mitt lösenord?",
        "ideal_answer": (
            "Guide 1 beskriver att användaren ska gå till "
            "portal.nordhamndigital.se/konto, välja Glömt lösenord eller Sätt om "
            "lösenord, verifiera sig med MFA och sedan skapa ett nytt starkt "
            "lösenord."
        ),
        "answer_keywords": [
            "portal.nordhamndigital.se/konto",
            "glömt lösenord",
            "mfa",
            "nytt starkt lösenord",
        ],
        "retrieval_keywords": [
            "guide 1", "lösenord", "återställ lösenord",
            "konto", "glömt lösenord", "mfa",
        ],
        "accepted_doc_refs": ["guide 1", "lösenord"],
        "soft_min_keyword_matches": 2,
    },
    {
        "query": "Hur får jag min skrivare?",
        "ideal_answer": (
            "Guide 8 beskriver att användaren ska skriva ut till "
            "FollowMe-Nordhamn, gå till en FollowMe-skrivare, blippa passerkort "
            "eller logga in med PIN och sedan frisläppa utskriften."
        ),
        "answer_keywords": [
            "followme-nordhamn", "followme", "passerkort", "pin", "frisläpp",
        ],
        "retrieval_keywords": [
            "guide 8", "followme", "skrivare", "passerkort",
            "pin", "säker utskrift",
        ],
        "accepted_doc_refs": ["guide 8", "followme"],
        "soft_min_keyword_matches": 2,
    },
    {
        "query": "Hur bokar jag mötesrum?",
        "ideal_answer": (
            "Guide 10 beskriver att användaren ska öppna Outlook och gå till "
            "Kalender, skapa ett nytt möte, välja rum från rumslistan och skicka "
            "bokningen."
        ),
        "answer_keywords": [
            "outlook", "kalender", "nytt möte", "rum", "rumslista",
        ],
        "retrieval_keywords": [
            "guide 10", "mötesrum", "outlook", "rumslista",
            "kalender", "teams-länk",
        ],
        "accepted_doc_refs": ["guide 10", "mötesrum"],
        "soft_min_keyword_matches": 2,
    },
    {
        "query": "Jag får felmeddelande när jag loggar in i Citrix, vad gör jag nu?",
        "ideal_answer": (
            "Guide 2 beskriver att användaren ska kontrollera att adressen "
            "citrix.nordhamndigital.se är rätt, starta om Citrix Workspace om "
            "sessionen stänger sig och logga ut helt och öppna appen på nytt om "
            "svart skärm uppstår."
        ),
        "answer_keywords": [
            "citrix.nordhamndigital.se", "citrix workspace", "starta om",
            "svart skärm", "logga ut",
        ],
        "retrieval_keywords": [
            "guide 2", "citrix", "workspace", "svart skärm",
            "sessionen stänger", "arbetsytan",
        ],
        "accepted_doc_refs": ["guide 2", "citrix"],
        "soft_min_keyword_matches": 2,
    },
    {
        "query": "Hur kommer jag in i min mail?",
        "ideal_answer": (
            "Guide 4 beskriver att användaren ska öppna Outlook, skriva in sin "
            "jobbadress, välja Jobb eller Skolkonto, ange lösenord och bekräfta "
            "inloggningen med MFA."
        ),
        "answer_keywords": [
            "outlook", "jobbadress", "jobb eller skolkonto", "lösenord", "mfa",
        ],
        "retrieval_keywords": [
            "guide 4", "outlook", "jobbmejl",
            "första inloggning", "kalender", "mfa",
        ],
        "accepted_doc_refs": ["guide 4", "outlook"],
        "soft_min_keyword_matches": 2,
    },
    {
        "query": "Hur ansluter jag till det trådlösa kontorsnätet?",
        "ideal_answer": (
            "Guide 7 beskriver att användaren ska välja Nordhamn-Staff i listan "
            "över trådlösa nätverk, logga in med företagskonto och lösenord och "
            "sedan testa att öppna intranätet eller Outlook."
        ),
        "answer_keywords": [
            "nordhamn-staff", "trådlösa nätverk", "företagskonto", "lösenord",
            "outlook",
        ],
        "retrieval_keywords": [
            "guide 7", "wifi", "nordhamn-staff", "trådlöst nätverk",
            "certifikat", "kontor",
        ],
        "accepted_doc_refs": ["guide 7", "nordhamn-staff", "wifi"],
        "soft_min_keyword_matches": 2,
    },
]


MODEL_CONFIGS = [
    {"provider": "ollama", "model_name": "llama3.2:3b"},
    # Tidigare tester visade att Gemini ofta gav snabbare svar och i flera fall
    # mer träffsäkra svar. Jag har ändå valt att inte ha med Gemini i den
    # aktiva jämförelsen här, eftersom gratisnivån gav quota-fel och gjorde
    # resultatet ojämnt och svårt att jämföra rättvist mot Ollama.
    # {"provider": "gemini", "model_name": "gemini-2.5-flash"},
]


K_VALUES = [2, 3, 4]
MIN_SCORE_VALUES = [0.30, 0.35, 0.40, 0.45]


NEGATION_WORDS = [
    "inte", "ej", "aldrig", "ingen", "inget", "inga",
    "nej", "fel", "varken", "utan",
]

UNCERTAINTY_PHRASES = [
    "vet inte", "framgår inte", "formgar inte",
    "förmodligen", "troligen", "kan vara", "oklart",
    "osäker", "inte säker", "kan inte svara",
    "hittar inte", "ingen information", "framkommer inte",
]


def prepare_chunks_for_evaluation(text: str) -> List[str]:
    """
    Bygger chunks och undviker att filterregler gör evalueringen missvisande.
    """
    chunks = chunk_text_by_paragraphs(text)
    if not chunks:
        raise ValueError("Dokumentet gav inga chunks att evaluera.")

    try:
        filtered_chunks = filter_low_value_chunks(chunks)
    except ValueError:
        print("Varning: låg-värdesfiltret tog bort alla chunks. Evalueringen körs utan filter.")
        return chunks

    retention_ratio = len(filtered_chunks) / len(chunks)

    if (
        len(filtered_chunks) < MIN_CHUNKS_AFTER_FILTER
        or retention_ratio < MIN_FILTER_RETENTION_RATIO
    ):
        print(
            "Varning: låg-värdesfiltret tog bort för stor del av dokumentet "
            f"({len(filtered_chunks)}/{len(chunks)} chunks kvar). "
            "Evalueringen körs utan filter."
        )
        return chunks

    return filtered_chunks


def normalize_text(text: str) -> str:
    """
    Normaliserar text för enkel jämförelse.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_answer_keywords(answer_keywords: List[str]) -> str:
    return ", ".join(answer_keywords)


def format_doc_refs(doc_refs: List[str]) -> str:
    return ", ".join(doc_refs)


def contains_negation_near_keyword(text: str, keyword: str, window: int = 6) -> bool:
    """
    Returnerar True om ett negationsord finns nära ett nyckelord.
    """
    words = normalize_text(text).split()
    keyword_norm = normalize_text(keyword)

    for i, word in enumerate(words):
        if word != keyword_norm:
            continue

        start = max(0, i - window)
        end = min(len(words), i + window + 1)
        surrounding = words[start:end]

        if any(neg in surrounding for neg in NEGATION_WORDS):
            return True

    return False


def contains_uncertainty(answer: str) -> bool:
    answer_norm = normalize_text(answer)
    return any(normalize_text(phrase) in answer_norm for phrase in UNCERTAINTY_PHRASES)


def count_keyword_matches(answer_keywords: List[str], answer: str) -> int:
    answer_norm = normalize_text(answer)
    return sum(1 for kw in answer_keywords if normalize_text(kw) in answer_norm)


def has_accepted_doc_ref(answer: str, accepted_doc_refs: List[str]) -> bool:
    if not accepted_doc_refs:
        return False

    answer_norm = normalize_text(answer)
    return any(normalize_text(doc_ref) in answer_norm for doc_ref in accepted_doc_refs)


def is_strict_correct(answer_keywords: List[str], answer: str) -> bool:
    """
    Strikt bedömning: alla nyckelord ska finnas och inga tydliga osäkerhets-
    eller negationsproblem får finnas.
    """
    if not answer or not answer.strip():
        return False

    if contains_uncertainty(answer):
        return False

    answer_norm = normalize_text(answer)
    all_present = all(normalize_text(kw) in answer_norm for kw in answer_keywords)
    if not all_present:
        return False

    any_negated = any(
        contains_negation_near_keyword(answer, kw)
        for kw in answer_keywords
    )
    return not any_negated


def is_soft_correct(
    answer_keywords: List[str],
    accepted_doc_refs: List[str],
    answer: str,
    min_keyword_matches: int = 2,
) -> bool:
    """
    Mjukare bedömning.

    Ett svar kan räcka om det:
    - pekar till rätt DOK-avsnitt
    - eller innehåller tillräckligt många relevanta nyckelord
    """
    if not answer or not answer.strip():
        return False

    if contains_uncertainty(answer):
        return False

    keyword_matches = count_keyword_matches(answer_keywords, answer)
    doc_ref_match = has_accepted_doc_ref(answer, accepted_doc_refs)

    return doc_ref_match or keyword_matches >= min_keyword_matches


def retrieval_hit_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    retrieval_keywords: List[str],
    top_n: int,
) -> bool:
    """
    Returnerar True om minst ett retrieval-nyckelord finns i top_n chunks.
    """
    if not retrieved_chunks:
        return False

    normalized_keywords = [normalize_text(kw) for kw in retrieval_keywords]

    for item in retrieved_chunks[:top_n]:
        chunk_text = normalize_text(item.get("text", ""))
        if any(kw in chunk_text for kw in normalized_keywords):
            return True

    return False


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Sammanfattar huvudmetrikerna för ett resultatset.
    """
    n = len(results)
    if n == 0:
        return {
            "retrieval_top1_pct": 0.0,
            "retrieval_top3_pct": 0.0,
            "retrieval_topk_pct": 0.0,
            "answer_soft_pct": 0.0,
            "answer_strict_pct": 0.0,
            "errors": 0,
            "avg_response_time_seconds": 0.0,
        }

    response_times = [r["response_time_seconds"] for r in results]

    return {
        "retrieval_top1_pct": sum(1 for r in results if r["retrieval_top1"]) / n * 100,
        "retrieval_top3_pct": sum(1 for r in results if r["retrieval_top3"]) / n * 100,
        "retrieval_topk_pct": sum(1 for r in results if r["retrieval_topk"]) / n * 100,
        "answer_soft_pct": sum(1 for r in results if r["answer_correct"]) / n * 100,
        "answer_strict_pct": sum(1 for r in results if r["answer_strict_correct"]) / n * 100,
        "errors": sum(1 for r in results if r["had_error"]),
        "avg_response_time_seconds": mean(response_times),
    }


def run_evaluation(
    chunks: List[str],
    chunk_embeddings,
    metadata: List[Dict[str, Any]],
    provider: str,
    model_name: str,
    k: int,
    min_score: float,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Kör alla testfall för en viss modellkonfiguration.
    """
    results = []

    for tc in TEST_CASES:
        query = tc["query"]
        had_error = False
        error_message = ""
        response_time_seconds = 0.0

        try:
            started_at = time.perf_counter()
            answer, retrieved, _ = answer_query(
                query=query,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                all_metadata=metadata,
                embed_query_func=embed_query,
                provider=provider,
                model_name=model_name,
                k=k,
                min_score=min_score,
            )
            response_time_seconds = time.perf_counter() - started_at
        except Exception as exc:
            had_error = True
            error_message = str(exc)
            print(f"\n  [FEL] Testfall kraschade: '{query}'")
            print(f"  Felmeddelande: {exc}")
            answer = f"[ERROR] {exc}"
            retrieved = []

        r_top1 = retrieval_hit_at_k(retrieved, tc["retrieval_keywords"], top_n=1)
        r_top3 = retrieval_hit_at_k(retrieved, tc["retrieval_keywords"], top_n=3)
        r_topk = retrieval_hit_at_k(retrieved, tc["retrieval_keywords"], top_n=k)

        a_hit_soft = is_soft_correct(
            tc["answer_keywords"],
            tc.get("accepted_doc_refs", []),
            answer,
            min_keyword_matches=tc.get("soft_min_keyword_matches", 2),
        )
        a_hit_strict = is_strict_correct(tc["answer_keywords"], answer)
        top_score = retrieved[0]["score"] if retrieved else 0.0

        results.append({
            "query": query,
            "ideal_answer": tc["ideal_answer"],
            "answer_keywords": format_answer_keywords(tc["answer_keywords"]),
            "accepted_doc_refs": format_doc_refs(tc.get("accepted_doc_refs", [])),
            "retrieval_keywords": ", ".join(tc["retrieval_keywords"]),
            "answer": answer.strip(),
            "top_score": top_score,
            "retrieval_top1": r_top1,
            "retrieval_top3": r_top3,
            "retrieval_topk": r_topk,
            "answer_correct": a_hit_soft,
            "answer_strict_correct": a_hit_strict,
            "retrieved_count": len(retrieved),
            "response_time_seconds": round(response_time_seconds, 3),
            "had_error": had_error,
            "error_message": error_message,
            "k": k,
            "min_score": min_score,
            "provider": provider,
            "model": model_name,
        })

        if debug:
            print(f"\n  [DEBUG] {query}")
            for item in retrieved[:3]:
                snippet = item.get("text", "").replace("\n", " ")[:100]
                print(
                    f"    Chunk {item.get('index', '?')} "
                    f"({item.get('score', 0):.3f}): {snippet}..."
                )
            print(f"  Svar         : {answer.strip()}")
            print(f"  Svarstid     : {response_time_seconds:.2f}s")
            print(f"  Retrieval@1  : {'✓' if r_top1 else '✗'}")
            print(f"  Retrieval@3  : {'✓' if r_top3 else '✗'}")
            print(f"  Retrieval@k  : {'✓' if r_topk else '✗'}")
            print(f"  Svar mjukt   : {'✓' if a_hit_soft else '✗'}")
            print(f"  Svar strikt  : {'✓' if a_hit_strict else '✗'}")
            if had_error:
                print(f"  ERROR        : {error_message}")

    return results


def save_results_to_csv(
    results: List[Dict[str, Any]],
    output_file: str = "evaluation_results.csv",
) -> None:
    """
    Sparar detaljresultat.
    """
    if not results:
        print("Inga resultat att spara.")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Detaljresultat sparade till: {output_path}")


def save_summary_to_csv(
    summary: List[Dict[str, Any]],
    output_file: str = "evaluation_summary.csv",
) -> None:
    """
    Sparar sammanfattning.
    """
    if not summary:
        print("Inga sammanfattningsresultat att spara.")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(summary[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"✓ Sammanfattning sparad till: {output_path}")


def grid_search(
    file_path: str,
    provider: str,
    model_name: str,
    results_file: str = "evaluation_results_grid.csv",
    summary_file: str = "evaluation_summary.csv",
    debug: bool = False,
) -> None:
    """
    Kör grid search över K och `min_score` för en modell.
    """
    print(f"\n{'=' * 70}")
    print("GRID SEARCH")
    print(f"Dokument      : {file_path}")
    print(f"Provider      : {provider}")
    print(f"Modell        : {model_name}")
    print(f"k-värden      : {K_VALUES}")
    print(f"min_score     : {MIN_SCORE_VALUES}")
    print(f"Kombinationer : {len(K_VALUES) * len(MIN_SCORE_VALUES)}")
    print(f"{'=' * 70}\n")

    text = load_document(file_path)
    chunks = prepare_chunks_for_evaluation(text)
    chunk_embeddings = embed_texts(chunks)
    metadata = [
        {"doc": file_path, "chunk_index": i, "category": "Test"}
        for i in range(len(chunks))
    ]

    print(f"Dokument inläst: {len(chunks)} chunks.\n")

    all_results = []
    summary = []

    for k, min_score in product(K_VALUES, MIN_SCORE_VALUES):
        print(f"  Testar k={k}, min_score={min_score} ...", end=" ", flush=True)

        results = run_evaluation(
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            metadata=metadata,
            provider=provider,
            model_name=model_name,
            k=k,
            min_score=min_score,
            debug=debug,
        )

        all_results.extend(results)
        metrics = summarize_results(results)

        print(
            f"R@1={metrics['retrieval_top1_pct']:.0f}%  "
            f"R@3={metrics['retrieval_top3_pct']:.0f}%  "
            f"R@k={metrics['retrieval_topk_pct']:.0f}%  "
            f"Svar mjukt={metrics['answer_soft_pct']:.0f}%  "
            f"Svar strikt={metrics['answer_strict_pct']:.0f}%  "
            f"Snittid={metrics['avg_response_time_seconds']:.2f}s  "
            f"Fel={metrics['errors']}"
        )

        summary.append({
            "provider": provider,
            "model": model_name,
            "k": k,
            "min_score": min_score,
            "retrieval_top1_pct": round(metrics["retrieval_top1_pct"], 1),
            "retrieval_top3_pct": round(metrics["retrieval_top3_pct"], 1),
            "retrieval_topk_pct": round(metrics["retrieval_topk_pct"], 1),
            "answer_accuracy_pct": round(metrics["answer_soft_pct"], 1),
            "answer_strict_accuracy_pct": round(metrics["answer_strict_pct"], 1),
            "avg_response_time_seconds": round(metrics["avg_response_time_seconds"], 3),
            "n_testfall": len(results),
            "n_errors": metrics["errors"],
        })

    print(f"\n{'=' * 70}")
    print("SAMMANFATTNING - ALLA KOMBINATIONER")
    print(
        f"{'k':<5} {'min_score':<11} {'R@1':>5} {'R@3':>5} {'R@k':>5} "
        f"{'Mjukt':>7} {'Strikt':>7} {'Tid':>7} {'Fel':>5}"
    )
    print("-" * 80)

    for row in summary:
        print(
            f"  {row['k']:<3}  {row['min_score']:<9}"
            f"  {row['retrieval_top1_pct']:>4.0f}%"
            f"  {row['retrieval_top3_pct']:>4.0f}%"
            f"  {row['retrieval_topk_pct']:>4.0f}%"
            f"  {row['answer_accuracy_pct']:>6.0f}%"
            f"  {row['answer_strict_accuracy_pct']:>6.0f}%"
            f"  {row['avg_response_time_seconds']:>6.2f}s"
            f"  {row['n_errors']:>4}"
        )

    best = max(
        summary,
        key=lambda row: (row["answer_accuracy_pct"], row["retrieval_top1_pct"]),
    )

    print(f"\n{'=' * 70}")
    print("BÄSTA KOMBINATION")
    print(f"{'=' * 70}")
    print(f"  k           = {best['k']}")
    print(f"  min_score   = {best['min_score']}")
    print(f"  Retrieval@1 = {best['retrieval_top1_pct']:.0f}%")
    print(f"  Retrieval@3 = {best['retrieval_top3_pct']:.0f}%")
    print(f"  Retrieval@k = {best['retrieval_topk_pct']:.0f}%")
    print(f"  Svar mjukt  = {best['answer_accuracy_pct']:.0f}%")
    print(f"  Svar strikt = {best['answer_strict_accuracy_pct']:.0f}%")
    print(f"  Snittid     = {best['avg_response_time_seconds']:.2f}s")
    print(f"  Fel         = {best['n_errors']}")

    save_results_to_csv(all_results, results_file)
    save_summary_to_csv(summary, summary_file)


def evaluate(
    file_path: str,
    provider: str,
    model_name: str,
    k: int = 5,
    min_score: float = 0.35,
    debug: bool = True,
    output_file: str = "evaluation_results.csv",
) -> List[Dict[str, Any]]:
    """
    Kör evaluering för en modell.
    """
    print(f"\n{'=' * 70}")
    print("STARTAR EVALUERING")
    print(f"Dokument   : {file_path}")
    print(f"Provider   : {provider}")
    print(f"Modell     : {model_name}")
    print(f"k          : {k}")
    print(f"min_score  : {min_score}")
    print(f"{'=' * 70}\n")

    text = load_document(file_path)
    chunks = prepare_chunks_for_evaluation(text)
    chunk_embeddings = embed_texts(chunks)
    metadata = [
        {"doc": file_path, "chunk_index": i, "category": "Test"}
        for i in range(len(chunks))
    ]
    print(f"Dokument inläst: {len(chunks)} chunks.\n")

    results = run_evaluation(
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        metadata=metadata,
        provider=provider,
        model_name=model_name,
        k=k,
        min_score=min_score,
        debug=debug,
    )

    metrics = summarize_results(results)

    print(f"\n{'=' * 70}")
    print("RAPPORT")
    print(f"{'=' * 70}")
    print(f"  Retrieval@1 accuracy : {metrics['retrieval_top1_pct']:.1f}%")
    print(f"  Retrieval@3 accuracy : {metrics['retrieval_top3_pct']:.1f}%")
    print(f"  Retrieval@k accuracy : {metrics['retrieval_topk_pct']:.1f}%")
    print(f"  Answer soft accuracy : {metrics['answer_soft_pct']:.1f}%")
    print(f"  Answer strict acc.   : {metrics['answer_strict_pct']:.1f}%")
    print(f"  Avg response time    : {metrics['avg_response_time_seconds']:.2f}s")
    print(f"  Errors               : {metrics['errors']}")
    print(f"{'=' * 70}\n")

    for row in results:
        soft_sym = "✓" if row["answer_correct"] else "✗"
        strict_sym = "✓" if row["answer_strict_correct"] else "✗"
        r1_sym = "✓" if row["retrieval_top1"] else "✗"
        r3_sym = "✓" if row["retrieval_top3"] else "✗"

        print(
            f"{soft_sym} Mjukt | {strict_sym} Strikt | "
            f"R@1:{r1_sym} R@3:{r3_sym} | "
            f"[{row['top_score']:.3f}] {row['query']}"
        )
        print(f"   Önskat svar   : {row['ideal_answer']}")
        print(f"   Nyckelord     : {row['answer_keywords']}")
        print(f"   Dok-ref       : {row['accepted_doc_refs']}")
        print(f"   Svarstid      : {row['response_time_seconds']:.2f}s")
        print(f"   Modellens svar: {row['answer']}")
        if row["had_error"]:
            print(f"   ERROR         : {row['error_message']}")
        print()

    print("Tolkning:")
    print("  Hög R@k men låg R@1              -> rätt info hämtas men rankas inte högst.")
    print("  Låg R@k                          -> retrieval hittar inte rätt info alls.")
    print("  Hög retrieval, låg answer acc    -> modellen använder kontexten dåligt.")
    print("  Höga mjuka svar men låga strikta -> svaret är ofta rimligt men inte exakt.")
    print("  Många fel                        -> något kraschar i pipeline eller modellanrop.")
    print("  Allt högt                        -> pipelinen fungerar bra.\n")

    save_results_to_csv(results, output_file)
    return results


def compare_models(
    file_path: str,
    model_configs: List[Dict[str, str]],
    k: int = 3,
    min_score: float = 0.30,
    debug: bool = False,
    output_file: str = "evaluation/evaluation_results.csv",
    summary_file: str = "evaluation/evaluation_summary.csv",
) -> List[Dict[str, Any]]:
    """
    Kör evaluering för flera modeller och sparar ett gemensamt underlag.
    """
    print(f"\n{'=' * 70}")
    print("MODELLJÄMFÖRELSE")
    print(f"Dokument   : {file_path}")
    print(f"k          : {k}")
    print(f"min_score  : {min_score}")
    print(f"Konfig     : {len(model_configs)} modeller")
    print(f"{'=' * 70}\n")

    text = load_document(file_path)
    chunks = prepare_chunks_for_evaluation(text)
    chunk_embeddings = embed_texts(chunks)
    metadata = [
        {"doc": file_path, "chunk_index": i, "category": "Test"}
        for i in range(len(chunks))
    ]
    print(f"Dokument inläst: {len(chunks)} chunks.\n")

    all_results = []
    summary_rows = []

    for model_config in model_configs:
        provider = model_config["provider"]
        model_name = model_config["model_name"]

        print(f"Kör {provider}/{model_name} ...")

        results = run_evaluation(
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            metadata=metadata,
            provider=provider,
            model_name=model_name,
            k=k,
            min_score=min_score,
            debug=debug,
        )

        all_results.extend(results)
        metrics = summarize_results(results)

        summary_rows.append({
            "provider": provider,
            "model": model_name,
            "k": k,
            "min_score": min_score,
            "retrieval_top1_pct": round(metrics["retrieval_top1_pct"], 1),
            "retrieval_top3_pct": round(metrics["retrieval_top3_pct"], 1),
            "retrieval_topk_pct": round(metrics["retrieval_topk_pct"], 1),
            "answer_accuracy_pct": round(metrics["answer_soft_pct"], 1),
            "answer_strict_accuracy_pct": round(metrics["answer_strict_pct"], 1),
            "avg_response_time_seconds": round(metrics["avg_response_time_seconds"], 3),
            "n_testfall": len(results),
            "n_errors": metrics["errors"],
        })

    print(f"\n{'=' * 70}")
    print("JÄMFÖRELSE")
    print(f"{'=' * 70}")
    print(
        f"{'Provider':<10} {'Modell':<20} {'R@1':>5} {'R@3':>5} "
        f"{'Mjukt':>7} {'Strikt':>7} {'Tid':>7} {'Fel':>5}"
    )
    print("-" * 80)

    for row in summary_rows:
        print(
            f"{row['provider']:<10} {row['model']:<20}"
            f" {row['retrieval_top1_pct']:>4.0f}%"
            f" {row['retrieval_top3_pct']:>4.0f}%"
            f" {row['answer_accuracy_pct']:>6.0f}%"
            f" {row['answer_strict_accuracy_pct']:>6.0f}%"
            f" {row['avg_response_time_seconds']:>6.2f}s"
            f" {row['n_errors']:>4}"
        )

    save_results_to_csv(all_results, output_file)
    save_summary_to_csv(summary_rows, summary_file)
    return all_results


if __name__ == "__main__":
    FILE_PATH = PROJECT_ROOT / "test_docs" / "nordhamn_digital_howto_nyanstalld.pdf"

    if len(MODEL_CONFIGS) == 1:
        evaluate(
            file_path=FILE_PATH,
            provider=MODEL_CONFIGS[0]["provider"],
            model_name=MODEL_CONFIGS[0]["model_name"],
            k=3,
            min_score=0.30,
            debug=True,
            output_file=EVALUATION_DIR / "evaluation_results.csv",
        )
    else:
        compare_models(
            file_path=FILE_PATH,
            model_configs=MODEL_CONFIGS,
            k=3,
            min_score=0.30,
            debug=False,
            output_file=EVALUATION_DIR / "evaluation_results.csv",
            summary_file=EVALUATION_DIR / "evaluation_summary.csv",
        )

    # Avkommentera om du vill grid-söka för en enskild modell:
    #
    # grid_search(
    #     file_path=FILE_PATH,
    #     provider="ollama",
    #     model_name="llama3.2:3b",
    #     results_file="evaluation/evaluation_results_grid.csv",
    #     summary_file="evaluation/evaluation_summary_grid.csv",
    #     debug=False,
    # )
