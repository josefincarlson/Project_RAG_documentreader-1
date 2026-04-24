# =========================
# test_llm_gemini.py — test av Gemini LLM-anrop
# =========================

# Används för att testa att anrop till Google Gemini fungerar korrekt innan vi integrerar det i hela systemet.
# Genom att ha denna testscript kunde jag snabbt verifiera att mina API-nycklar och inställningar fungerade som de skulle, vilket underlättade felsökning och utveckling av Gemini-integrationen i appen.
# I det här testet försöker jag generera ett enkelt svar från Gemini-modellen "gemini-2.5-flash" och skriver ut svaret i konsolen för att verifiera att allt fungerar som det ska.
# Jag använder också den generella generate_response-funktionen från llm.py för att testa att den fungerar med Gemini-provider och att jag får ett svar som jag kan använda i appen.


from src.llm import generate_response

response = generate_response(
    provider="gemini",
    model_name="gemini-2.5-flash",
    system_prompt="Du svarar alltid kort på svenska.",
    user_prompt="Säg hej"
)

print(response)