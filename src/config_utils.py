# =========================
# Konfiguration och policy
# =========================
#
# Samlar uppslag mot config och säkerhetspolicy på ett ställe,
# så att resten av appen slipper läsa config direkt.


from src.config import APP_SECURITY_SETTINGS, LLM_SETTINGS, EMBEDDING_SETTINGS


# =========================
# Interna LLM-uppslag
# =========================
#
# Läser ut providers och modeller från config och ger tydliga fel om något saknas.

def _get_llm_providers() -> dict:
    """
    Returnerar alla konfigurerade LLM-providers.
    """
    providers = LLM_SETTINGS.get("providers", {})

    if not providers:
        raise ValueError(
            "LLM_SETTINGS['providers'] är tom eller saknas. Minst en LLM-provider måste vara konfigurerad."
        )

    return providers


def _get_llm_provider_entry(provider: str) -> dict:
    """
    Hämtar config för en provider och validerar att namnet är användbart.
    Ger tidiga fel om provider saknas eller är felstavad.
    """
    if provider is None:
        raise ValueError("provider saknas.")

    if not isinstance(provider, str):
        raise TypeError("provider måste vara en sträng.")

    if not provider.strip():
        raise ValueError("provider är tom.")

    providers = _get_llm_providers()

    if provider not in providers:
        raise ValueError(
            f"Provider '{provider}' finns inte i LLM_SETTINGS['providers']."
        )

    return providers[provider]


def _get_llm_models_map(provider: str) -> dict:
    """
    Returnerar modellmappningen för vald provider.
    """
    provider_entry = _get_llm_provider_entry(provider)
    models = provider_entry.get("models", {})

    if not models:
        raise ValueError(
            f"Provider '{provider}' har inga modeller konfigurerade i LLM_SETTINGS['providers']."
        )

    return models


# =========================
# Säkerhetsläge och policy
# =========================
#
# Hjälper appen att avgöra vad som är tillåtet i lokalt läge respektive hybridläge.

def get_security_mode() -> str:
    """
    Returnerar aktivt säkerhetsläge.
    Tillåtna värden: 'local_only', 'hybrid'
    """
    mode = APP_SECURITY_SETTINGS.get("mode", "local_only")

    if mode not in {"local_only", "hybrid"}:
        raise ValueError(
            f"Ogiltigt säkerhetsläge i APP_SECURITY_SETTINGS['mode']: {mode}"
        )

    return mode


def should_warn_on_cloud_usage() -> bool:
    """
    Returnerar True om appen ska visa varningar för molnbaserade modeller.
    """
    return bool(APP_SECURITY_SETTINGS.get("warn_on_cloud_usage", True))


def is_cloud_llm_allowed() -> bool:
    """
    Returnerar True om molnbaserade LLM:er är tillåtna enligt aktiv policy.
    """
    return get_security_mode() == "hybrid"


def is_cloud_embeddings_allowed() -> bool:
    """
    Returnerar True om molnbaserade embeddings är tillåtna enligt aktiv policy.
    """
    return get_security_mode() == "hybrid"


# =========================
# Embedding-konfiguration
# =========================
#
# Läser ut den aktiva embedding-modellen och kontrollerar att den är tillåten.

def get_active_embedding_config() -> dict:
    """
    Hämtar konfigurationen för den aktiva embedding-modellen.
    
    Embedding-modellen styrs i config och är inte valbar i UI.
    """
    model_key = EMBEDDING_SETTINGS.get("active_model")
    models = EMBEDDING_SETTINGS.get("models", {})

    if not models:
        raise ValueError(
            "EMBEDDING_SETTINGS['models'] är tom eller saknas. Minst en embedding-modell måste vara konfigurerad."
        )

    if not model_key:
        raise ValueError("EMBEDDING_SETTINGS saknar 'active_model'.")

    if model_key not in models:
        raise ValueError(
            f"Aktiv embedding-modell '{model_key}' finns inte i EMBEDDING_SETTINGS['models']."
        )

    model_config = models[model_key]

    return {
        "model_key": model_key,
        "provider": model_config["provider"],
        "model_name": model_config["model_name"],
        "query_prefix": model_config.get("query_prefix", ""),
        "document_prefix": model_config.get("document_prefix", ""),
        "execution_mode": model_config.get("execution_mode", "local"),
        "requires_api_key": model_config.get("requires_api_key", False),
        "show_cloud_warning": model_config.get("show_cloud_warning", False),
        "normalize_embeddings": EMBEDDING_SETTINGS.get("normalize_embeddings", True),
    }


def validate_embedding_policy() -> dict:
    """
    Validerar att den aktiva embedding-modellen är tillåten i aktuellt säkerhetsläge.
    Det är viktigt eftersom embedding-modellen styrs via config och inte kan väljas bort i UI.
    """
    config = get_active_embedding_config()

    if (
        config["execution_mode"] == "cloud"
        and not is_cloud_embeddings_allowed()
    ):
        raise ValueError(
            f"Embedding-modellen '{config['model_key']}' är molnbaserad och tillåts inte i säkerhetsläget '{get_security_mode()}'."
        )

    return config


# =========================
# Hjälpfunktioner för val av LLM-provider och standardmodell
# =========================
#
# Denna del innehåller hjälpfunktioner för att läsa ut tillgängliga
# LLM-providers och modeller från konfigurationen. Funktionerna används för
# att hämta provider-namn, filtrera fram providers som är tillåtna enligt
# aktiv säkerhetspolicy samt läsa ut och validera standardval för provider
# och modell.

def get_llm_provider_names() -> list[str]:
    """
    Returnerar alla tillgängliga LLM-providers.
    """
    return list(_get_llm_providers().keys())


def get_allowed_llm_provider_names() -> list[str]:
    """
    Returnerar de LLM-providers som är tillåtna i aktuellt säkerhetsläge.
    """
    allowed_providers = [
        provider
        for provider in get_llm_provider_names()
        if not (
            get_llm_provider_config(provider)["execution_mode"] == "cloud"
            and not is_cloud_llm_allowed()
        )
    ]

    if not allowed_providers:
        raise ValueError(
            "Inga LLM-providers är tillåtna i aktuellt säkerhetsläge."
        )

    return allowed_providers


def get_llm_models_for_provider(provider: str) -> list[str]:
    """
    Returnerar alla modeller för vald provider.
    """
    return list(_get_llm_models_map(provider).keys())


def get_default_llm_provider() -> str:
    """
    Hämtar default-provider för LLM.
    """
    provider = LLM_SETTINGS.get("default_provider")

    if not provider:
        raise ValueError("LLM_SETTINGS saknar 'default_provider'.")

    return provider


def get_default_llm_model(provider: str) -> str:
    """
    Hämtar defaultmodell för vald provider och validerar att den finns.
    """
    if not provider:
        raise ValueError("provider saknas.")

    default_models = LLM_SETTINGS.get("default_model_per_provider", {})

    if provider not in default_models:
        raise ValueError(f"Defaultmodell saknas för provider '{provider}'.")

    model_name = default_models[provider]
    provider_models = _get_llm_models_map(provider)

    if model_name not in provider_models:
        raise ValueError(
            f"Defaultmodellen '{model_name}' finns inte bland modellerna för provider '{provider}'."
        )

    return model_name


# =========================
# Hjälpfunktioner för LLM-konfiguration
# =========================
#
# Denna del innehåller hjälpfunktioner för att läsa ut och sammanställa
# konfiguration för LLM-providers och modeller. Funktionerna används för att
# hämta provider-inställningar, modellspecifika inställningar och den
# fullständiga aktiva LLM-konfigurationen utifrån val i config eller
# angivna parametrar.

def get_llm_provider_config(provider: str) -> dict:
    """
    Hämtar provider-konfiguration för vald LLM-provider.
    """
    provider_config = _get_llm_provider_entry(provider)
    models = _get_llm_models_map(provider)

    return {
        "provider": provider,
        "display_name": provider_config.get("display_name", provider),
        "execution_mode": provider_config.get("execution_mode", "local"),
        "requires_api_key": provider_config.get("requires_api_key", False),
        "show_cloud_warning": provider_config.get("show_cloud_warning", False),
        "models": models,
    }


def get_llm_model_settings(
    provider: str,
    model_name: str
) -> dict:
    """
    Hämtar inställningar för en specifik modell hos vald provider.
    """
    if not model_name:
        raise ValueError("model_name saknas.")

    provider_models = _get_llm_models_map(provider)

    if model_name not in provider_models:
        raise ValueError(
            f"Modellen '{model_name}' finns inte för provider '{provider}'."
        )

    return provider_models[model_name]


def get_active_llm_config(
    provider: str | None = None,
    model_name: str | None = None
) -> dict:
    """
    Hämtar komplett aktiv LLM-konfiguration.
    Om provider eller model_name inte skickas in används defaultvärden från config.
    """
    selected_provider = provider or get_default_llm_provider()
    selected_model = model_name or get_default_llm_model(selected_provider)

    provider_config = _get_llm_provider_entry(selected_provider)
    model_settings = get_llm_model_settings(selected_provider, selected_model)

    return {
        "provider": selected_provider,
        "model_name": selected_model,
        "display_name": provider_config.get("display_name", selected_provider),
        "execution_mode": provider_config.get("execution_mode", "local"),
        "requires_api_key": provider_config.get("requires_api_key", False),
        "show_cloud_warning": provider_config.get("show_cloud_warning", False),
        "model_settings": model_settings,
    }


# =========================
# Hjälpfunktioner för LLM-policy
# =========================
#
# Denna del innehåller hjälpfunktioner för att validera att vald eller aktiv
# LLM-konfiguration är tillåten enligt appens säkerhetspolicy. Funktionen
# används för att kontrollera om en molnbaserad provider får användas i det
# aktuella säkerhetsläget innan modellen anropas.

def validate_llm_policy(
    provider: str | None = None, 
    model_name: str | None = None
) -> dict:
    """
    Kontrollerar om vald eller aktiv LLM-konfiguration är tillåten enligt säkerhetspolicyn.
    """
    config = get_active_llm_config(
        provider=provider, 
        model_name=model_name
    )

    if (
        config["execution_mode"] == "cloud"
        and not is_cloud_llm_allowed()
    ):
        raise ValueError(
            f"LLM-provider '{config['provider']}' är molnbaserad och tillåts inte i säkerhetsläget '{get_security_mode()}'."
        )

    return config
