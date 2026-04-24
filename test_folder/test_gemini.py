# =========================
# test_gemini.py — test av Google Gemini API
# =========================

# Användes för att testa att jag kunde ansluta till Google Gemini API och få ett svar innan jag implementerade det i appen.
# Genom att ha denna testscript kunde jag snabbt verifiera att mina API-nycklar och inställningar fungerade som de skulle, vilket underlättade felsökning och utveckling av Gemini-integrationen i appen.
# I det här testet försöker jag generera ett enkelt svar från Gemini-modellen "gemini-2.5-flash" och skriver ut svaret i konsolen för att verifiera att allt fungerar som det ska.

import os
print("Startar test...")

from google import genai
print("google.genai importerad")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY saknas i miljövariablerna.")

print("API-nyckel hittad")

client = genai.Client(api_key=api_key)
print("Client skapad")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Svara bara med ordet: hej"
)

print("Svar mottaget")
print(response.text)