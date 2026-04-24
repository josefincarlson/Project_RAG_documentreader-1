# =========================
# config.py - Samlade inställningar för appen
# =========================
#
# Denna fil innehåller appens centrala konfiguration, till exempel
# säkerhetspolicy, LLM-inställningar, embedding-modeller,
# chunking-parametrar och UI-val. Om du vill ändra standardinställningar,
# tillgängliga modeller eller andra justerbara värden är detta rätt fil.


# =========================
# Appens säkerhetspolicy
# =========================
#
# "mode" styr om appen bara får använda lokala modeller eller även
# molnbaserade modeller.
# - local_only: bara lokala modeller tillåts
# - hybrid: både lokala och molnbaserade modeller tillåts
#
# "warn_on_cloud_usage" styr om varningar ska visas för molnbaserade modeller.
# Säkerhetsläget påverkar också vilka providers och modeller som visas i appen.

APP_SECURITY_SETTINGS = {
    "mode": "hybrid",  # local_only | hybrid
    "warn_on_cloud_usage": True,
}


# =========================
# LLM-inställningar
# =========================
#
# Styr vilka LLM-providers och modeller som finns tillgängliga i appen.
# Inställningarna används i app_streamlit.py för val i UI:t och i llm.py
# när frågor skickas till den valda modellen.
# Här anges också standardval vid appstart och inställningar för varje
# provider och modell, till exempel körläge, API-nyckel och temperatur.

LLM_SETTINGS = {
    "default_provider": "ollama",
    "default_model_per_provider": {
        "ollama": "llama3.2:3b",
        "gemini": "gemini-2.5-flash",
    },
    "providers": {
        "ollama": {
            "display_name": "Ollama",
            "execution_mode": "local",
            "requires_api_key": False,
            "show_cloud_warning": False,
            "models": {
                "llama3.2:3b": {
                    "temperature": 0.2,
                },
                "llama3.1:8b": {
                    "temperature": 0.2,
                },
#                "mistral": {               # Testad, presterade inte tillräckligt bra
#                    "temperature": 0.2,
#                },
#                "gemma2:9b": {             # Testad, presterade inte tillräckligt bra
#                    "temperature": 0.2,    
#                },
#                "gemma3:1b": {             # Testad, presterade inte tillräckligt bra
#                    "temperature": 0.2,    
#                },
#                "qwen2.5:1.5b": {          # Testad, presterade inte tillräckligt bra
#                    "temperature": 0.2,
#                },
            },
        },
        "gemini": {
            "display_name": "Google Gemini",
            "execution_mode": "cloud",
            "requires_api_key": True,
            "show_cloud_warning": True,
            "models": {
                "gemini-2.5-flash": {
                    "temperature": 0.2,
                },
            },
        },
    },
}


# =========================
# Chunking-inställningar
# =========================
#
# Styr hur text delas upp i chunks innan embeddings och sökning.
# Dessa inställningar används i chunking.py för att bestämma storlek,
# längdenhet och filtrering av lågmässiga eller för korta textbitar.

CHUNKING_SETTINGS = {
    "length_unit": "character",   # character | token_estimate
    "max_chunk_size": 500,
    "min_chunk_size": 80,
    "overlap_size": 0,
    "low_value_filter_min_length_per_extension": {
        ".txt": 120,
        ".docx": 120,
        ".pdf": 120,
        ".pptx": 40,
        ".csv": 20,
        ".xlsx": 20,
    },
}


# =========================
# Embedding-inställningar
# =========================
#
# Styr vilken embedding-modell som används i appen och vilka modeller som
# finns tillgängliga. Inställningarna används i embeddings.py när text
# omvandlas till vektorer för sökning.
#
# "normalize_embeddings" styr om embeddings normaliseras innan jämförelse.
# "active_model" anger vilken modell som används globalt och kan inte ändras
# i UI:t av användare.
#
# Molnbaserade embeddings kräver giltig API-nyckel och måste följa appens
# säkerhetsläge.

EMBEDDING_SETTINGS = {
    "active_model": "kb_lab_sv",
    "normalize_embeddings": True,
    "models": {
        "kb_lab_sv": {
            "provider": "sentence_transformers",
            "model_name": "KBLab/sentence-bert-swedish-cased",
            "query_prefix": "",
            "document_prefix": "",
            "execution_mode": "local",
            "requires_api_key": False,
            "show_cloud_warning": False,
        },
        "e5_multilingual": {
            "provider": "sentence_transformers",
            "model_name": "intfloat/multilingual-e5-large-instruct",
            "query_prefix": (
                "Instruct: Given a web search query, retrieve relevant passages "
                "that answer the query.\nQuery: "
            ),
            "document_prefix": "",
            "execution_mode": "local",
            "requires_api_key": False,
            "show_cloud_warning": False,
        },
        "bge_m3": {
            "provider": "sentence_transformers",
            "model_name": "BAAI/bge-m3",
            "query_prefix": "",
            "document_prefix": "",
            "execution_mode": "local",
            "requires_api_key": False,
            "show_cloud_warning": False,
        },
    },
}


# =========================
# Dokumentladdningsinställningar
# =========================
#
# Styr hur dokument laddas in och hur text extraheras från olika filtyper.
# Inställningarna används i loaders.py för att hantera stödda filformat,
# hjälptester i UI:t och filtypsspecifika läsinställningar.

DOCUMENT_LOADING_SETTINGS = {
    "supported_extensions": ["pdf", "txt", "docx", "pptx", "csv", "xlsx"],
    "upload_help_text": "Stöder PDF (ej inskannade bilder), TXT, DOCX, PPTX, CSV och XLSX",
    "txt": {
        "encoding": "utf-8",
    },
    "pdf": {
        "encodings_to_try": [
            ("latin-1", "cp1252"),          # Vanligast för svenska 90-tal/00-tal (Word, SOU)
            ("latin-1", "utf-8"),           # Moderna PDF:er med fel metadata
            ("latin-1", "iso-8859-1"),      # Äldre Unix/Linux-genererade PDF:er
            ("latin-1", "iso-8859-15"),     # ISO med euro-tecken
            ("cp1252", "utf-8"),            # Windows-dokument felkodade som UTF-8
        ],
    },
    "csv": {
        "encoding": "utf-8-sig",
        "sniffer_delimiters": ",;",
        "sample_size": 2048,
    },
}


# =========================
# RAG-inställningar
# =========================
#
# Styr hur RAG-pipelinen arbetar, till exempel hur många chunks som hämtas
# vid sökning och vilken lägsta similarity-score som krävs.
# Inställningarna används i rag_pipeline.py.

RAG_SETTINGS = {
    "retrieval": {
        "top_k": 8,
        "min_score": 0.35,
    },
    "summary_retrieval": {
        "enabled": True,
        "top_k": 12,
        "min_score": 0.20,
    },
    "prompting": {
        "context_separator": "\n\n---\n\n",
    },
}


# =========================
# UI-inställningar
# =========================
#
# Styr inställningar för användargränssnittet, till exempel vilka
# dokumentkategorier som finns tillgängliga i appen.
# Inställningarna används i ui_components.py.

UI_SETTINGS = {
    "category_options": [
        "HR",
        "Ekonomi",
        "Facility",
        "Sales",
        "Policys",
        "Manuals/Wiki",
        "IT",
        "Security",
        "Development",
        "Övrigt",
    ],
}

