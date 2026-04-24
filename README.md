# 📄 Intelligent dokumentassistent för företag med RAG

**Version:** `v1.2026.04.24`

En RAG-baserad dokumentassistent som låter användare ladda upp interna dokument och ställa frågor om innehållet i ett chattgränssnitt. Systemet stöder både lokal körning via Ollama och molnbaserad körning via Gemini API, beroende på vald konfiguration.

Huvudfokus är lokal körning, så att känslig information kan stanna i den egna miljön. Gemini finns med som demonstrationsläge för att visa att lösningen även kan stödja molnbaserade modeller.


---


## 📌 Syfte
Syftet med projektet är att undersöka hur en RAG-baserad chattbot kan användas i en affärstillämpning för att effektivisera informationssökning i företagsdokument.

Lösningen är tänkt för exempelvis:

- HR-policys
- IT-manualer
- how-to-dokument
- wiki-sidor
- onboardingmaterial
- interna handböcker och rutiner

Målet är att göra information mer lättillgänglig och minska tiden som medarbetare lägger på att leta i dokument manuellt.


---


## 🚀 Funktioner
- Ladda upp dokument i flera filformat: PDF, TXT, DOCX, PPTX, CSV, XLSX
- Ställa frågor om dokumentinnehåll i ett chattgränssnitt
- Välja lokal eller molnbaserad språkmodell
- Säkerhetslägen för lokal eller hybrid körning
- Varningar vid användning av molnbaserade modeller
- Kontroll av API-nyckel vid Gemini-användning
- Kategorisering av uppladdade dokument
- Källreferenser och källchunks i gränssnittet
- Justerbart antal chunks och similarity-score
- Enkel feedback via 👍 / 👎
- Separat evaluering av retrieval och svarskvalitet via `evaluation.py`


---


## 🏗️ Arkitektur
Systemet bygger på RAG-principen i följande steg:

```text
Dokument → Chunking → Embeddings → Vector search → LLM → Svar
```

1. **Laddning** — Dokumentet läses in och text extraheras beroende på filformat.
2. **Chunking** — Texten delas upp i mindre textstycken baserat på naturliga styckesgränser. I den nuvarande versionen används teckenbaserad chunkstorlek, utan overlap, eftersom det gav stabilare retrieval på projektets testdokument än den tidigare större token-estimerade chunkingen med overlap.
3. **Embeddings** — Varje chunk omvandlas till en vektorrepresentation med en embedding-modell.
4. **Retrieval** — Vid en fråga jämförs frågans embedding med dokumentens embeddings via cosine similarity för att hitta de mest relevanta textstyckena.
5. **Generering** — De mest relevanta chunkarna skickas vidare som kontext till vald språkmodell, som genererar ett svar utifrån det hämtade underlaget.


---


## 📁 Projektstruktur
Project_RAG_documentreader/
│
├── app_streamlit.py                  # Streamlit-appens huvudfil
├── requirements.txt                  # Projektets beroenden
├── README.md                         # Projektdokumentation
├── .env.example                      # Exempel på miljövariabler
├── .gitignore                        # Filer och mappar som inte ska versionshanteras
│
├──evaluation/
│   ├── evaluation.py                 # Automatisk utvärdering av retrieval och svar
│   ├── evaluation_results.csv        # Resultat från senaste evalueringen
│   └──evaluation_visualisering.ipynb # Notebook för visualisering av resultaten
│
├── src/
│   ├── config.py                     # Samlade inställningar för appen
│   ├── config_utils.py               # Hjälpfunktioner för config, policy och validering
│   ├── chunking.py                   # Delar upp text i chunks och filtrerar lågkvalitativ text
│   ├── embeddings.py                 # Skapar vektorrepresentationer för text och frågor
│   ├── llm.py                        # Anropar vald LLM-provider (Ollama eller Gemini)
│   ├── loaders.py                    # Läser in dokument och extraherar text
│   ├── rag_pipeline.py               # Hanterar hela RAG-flödet från fråga till svar
│   ├── retriever.py                  # Retrieval via embeddings och cosine similarity
│   └── ui_components.py              # Återanvändbara UI-komponenter för Streamlit
│
├── test_docs/                        # Testdokument
└── test_folder/                      # Testscript


---


## ⚙️ Installation
### Krav
- Python 3.10+
- Ollama installerat och igång för lokal körning

### 1. Klona projektet
```bash
git clone https://github.com/josefin_carlson/Project_RAG_documentreader.git
cd Project_RAG_documentreader
```
### 2. Installera beroenden
```bash
pip install -r requirements.txt
```
### 3A. Ladda ner en lokal LLM-modell via Ollama
```bash
ollama pull llama3.2:3b
```
> Tillgängliga modeller i appen: `llama3.2:3b`, `llama3.1:8b`
> Ladda ner den eller de modeller du vill använda:
> ```bash
> ollama pull llama3.2:3b    # Snabbast, bra för laptops med begränsat RAM
> ollama pull llama3.1:8b    # Standard, bra balans mellan hastighet och kvalitet  
> ```
> Du behöver bara ladda ner de modeller du faktiskt tänker använda.

### 3B. Online-modell via Gemini API
För att använda Gemini behöver en giltig API-nyckel anges i en `.env`-fil i projektets rotmapp.

1. Kopiera `.env.example` till `.env`
2. Fyll i din API-nyckel för Gemini

Exempel:

```env
GEMINI_API_KEY=din_api_nyckel
```

### 4. Starta appen
```bash
streamlit run app_streamlit.py
```

---


## 📂 Filformat som stöds
| Status | Filformat |
|--------|-----------|
| ✅ Stöds nu | PDF, TXT, DOCX, PPTX, CSV, XLSX |
| 🔜 Planerat | HTML, HTM, MD, PY, IPYNB, JSON, XML |

> OBS: PDF-filer måste innehålla text — inskannade PDF:er (bilder) stöds inte ännu.


---


## 🔒 Säkerhetsläge och drift
Appen har stöd för olika säkerhetslägen för att tydligare kunna styra vilka modeller som får användas.

**Local only**: endast lokala LLM- och embedding-modeller tillåts
**Hybrid**: både lokala och molnbaserade modeller tillåts

Detta används för att:

- styra vilka providers och modeller som är valbara i appen
- blockera molnalternativ i strikt lokalt läge
- visa tydliga varningar när molnbaserade modeller används
- kontrollera att nödvändiga API-nycklar finns när exempelvis Gemini väljs

För verksamheter med höga sekretesskrav rekommenderas lokal körning.


---


## 📊 Utvärdering
Projektet innehåller en separat evaluation.py för att utvärdera systemet.

Evalueringen mäter både:
- retrieval-kvalitet
- svarskvalitet

Detta är viktigt eftersom systemet kan hämta rätt chunks men ändå ge ett svagt eller ofullständigt svar.

Evalueringen utgår från samma grundidé som i kurslitteraturen: att testa frågor mot en RAG-pipeline, kontrollera om relevant kontext hämtas och därefter bedöma om modellen ger ett användbart svar. I min lösning har jag byggt vidare på detta genom att separera retrieval och svarsgenerering i utvärderingen.

För varje testfråga finns ett önskat svar, relevanta nyckelord och förväntade dokumentreferenser. Evalueringen mäter därför både om rätt information hämtas (`retrieval_top1`, `retrieval_top3`, `retrieval_topk`) och om svaret kan godkännas. Svar bedöms på två nivåer:
- **mjuk bedömning**: svaret godkänns om det har rätt innebörd eller tydligt hänvisar till rätt dokument/avsnitt
- **strikt bedömning**: svaret måste ligga närmare det förväntade innehållet och matcha fler centrala nyckelord

Jag har också lagt till svarstid per modell för att kunna jämföra lokal och molnbaserad körning. Det gör evalueringen mer relevant för en verklig affärstillämpning, eftersom det inte bara handlar om korrekthet utan också om användarupplevelse, prestanda och avvägningen mellan dataskydd och svarskvalitet.

På så sätt följer evalueringen bokens upplägg, men är utökad för att bättre passa en komplett RAG-applikation där både retrieval, svarskvalitet och prestanda är viktiga.

Exempel på lärdomar från testning/insikter:
- retrieval och svarsgenerering behöver bedömas separat.
- systemet fungerar bättre på löpande text än på tabellstrukturerad data.
- Gemini presterade bättre än Ollama på XLSX- och CSV-liknande innehåll.
- chunking-inställningar påverkade resultatet tydligt, och mindre teckenbaserade chunks gav i den slutliga lösningen bättre retrieval än större token-estimerade chunks.
- en konfiguration med 2 chunks och minsta similarity-score 0.45 gav i flera tester renare kontext, men den bästa balansen berodde också på dokumenttyp och chunking-strategi.
- källvisning ökade transparensen och gjorde svaren lättare att verifiera.
- systemet fungerade bäst på tydliga och avgränsade frågor.
- kombinerade frågor, till exempel både sammanfattning och källplats, gav ibland ofullständiga svar.
- mindre modeller gav ibland för generella svar eller svarade “Det vet jag inte” trots relevant retrieval.
Kör evalueringen med:

```bash
python evaluation.py
```

Exempel från senaste evalueringen:
```
Retrieval@1 accuracy : 85.7%
Retrieval@3 accuracy : 100.0%
Retrieval@k accuracy : 100.0%
Answer soft accuracy : 100.0%
Answer strict acc.   : 28.6%
Avg response time    : 57.70s
Errors               : 0
```

Detaljresultaten sparas i `evaluation_results/evaluation_results.csv` och kan visualiseras vidare i notebooken `evaluation_results/evaluation_visualisering.ipynb`.

Tolkning av resultatet
Resultatet visar att retrieval fungerar mycket bra i testuppsättningen. Den mjuka svarskvaliteten är hög, men den strikta bedömningen är lägre eftersom modellen inte alltid formulerar svaren exakt enligt den ideala referensen. Detta visar varför både retrieval och svarsgenerering behöver utvärderas separat.


---


## ⚠️ Kända begränsningar
- **Datasekretess** — lokal körning via Ollama rekommenderas för känsliga dokument. Vid Gemini-läge skickas delar av dokumentinnehållet till extern tjänst. 
- **Stora PDF-filer** - kan leda till längre bearbetningstid och högre minnesanvändning.
- **Inskannade PDF-filer** - (bilder istället för text) stöds inte utan OCR.
- **Komplex layout** - i dokument (tabeller, flera kolumner) kan påverka textutvinningens kvalitet.
- **Prestanda** - lokal körning utan GPU ger längre svarstider. Produktionsmiljö bör ha dedikerad hårdvara. 
- **Hallucineringsrisk** — mindre modeller (3b–8b parametrar) kan ge felaktiga svar. Källchunks visas alltid så att användaren kan verifiera mot originalet.
- **Ingen persistent lagring** — dokument och embeddings lagras enbart i minnet under sessionen och försvinner vid omstart. 
- **DOCX-tabeller** — tabellinnehåll extraheras efter löpande text oavsett var i dokumentet tabellen finns. För de flesta dokument spelar detta ingen roll, men kan påverka retrieval i dokument där tabeller och löpande text är tätt blandade.
- **XLSX- och CSV-frågor** — tabellstrukturerad data kan vara svårare för modellen att tolka än löpande text. Vid testning kunde retrieval hämta rätt chunks från budgetfiler, men den lokala modellen via Ollama gav ibland allmänna eller ofullständiga svar trots korrekt källa.
- **Modellskillnader beroende på datatyp** — vid frågor mot tabellbaserat innehåll presterade Gemini bättre än Ollama i både svarskvalitet och svarstid. Det visar att resultatet inte bara påverkas av retrieval, utan också av språkmodellens förmåga att tolka strukturerad data.
- **Gemini i gratisnivå** — Gemini testades som molnalternativ och gav i flera fall snabbare svar och ibland bättre träffsäkerhet än den lokala modellen. Under fortsatt testning uppstod dock både `503 UNAVAILABLE` och `429 RESOURCE_EXHAUSTED`, vilket gjorde resultatet ojämnt. Därför valde jag att fokusera på Ollama som huvudspår i den slutliga lösningen.
- **Gemini-embeddings** — stöd för Gemini-embeddings är planerat men inte färdigimplementerat i denna version. Retrieval använder i nuläget lokala embedding-modeller.

> Källvisningen i gränssnittet bygger i stor utsträckning på metadata och UI-logik, till exempel dokumentnamn, chunk-index och sidmarkörer som extraheras och visas tillsammans med svaret. Modellen får också källkontext i prompten, men den exakta presentationen av källor i appen stöds av gränssnittet.


---


## 💼 Affärspotential och etiska perspektiv
Den här applikationen är framtagen som en dokumentassistent för företag, där medarbetare kan ladda upp interna dokument och ställa frågor om innehållet i ett chattgränssnitt. Ett viktigt mål med lösningen är att känslig information inte ska lämna företaget. Därför är huvudfokus lokal körning via Ollama, och appen har säkerhetsinställningar som gör det möjligt att styra om endast lokala modeller ska användas. Tanken är att appen vid en verklig driftsättning hos företag ska kunna köras i ett strikt lokalt läge för att minska risken att konfidentiell data skickas till externa tjänster.

Appen är byggd för att kunna användas som en produkt och inte bara som ett testprojekt. Användaren kan ladda upp dokument, märka dem med kategori och även ändra kategori i efterhand. Strukturen med separata moduler och konfigurationsfiler gör det enkelt att vidareutveckla lösningen, till exempel genom att byta LLM-modell, embedding-modell eller justera säkerhetsinställningar utan att behöva skriva om hela applikationen.

Affärsvärdet ligger i att minska tiden som medarbetare lägger på att leta efter information i dokument, policys, onboardingmaterial, interna handböcker och andra företagsdokument. Lösningen kan bidra till snabbare informationssökning, bättre åtkomst till intern kunskap och mer enhetlig användning av företagets dokumentation. Källchunks och källreferenser visas i gränssnittet tillsammans med svaret, med dokumentnamn och stycke, och ibland sidnummer när sådan information finns tillgänglig.

För att kunna säljas som en färdig produkt behöver lösningen vidareutvecklas. En viktig del är att införa persistent lagring av dokument och embeddings, exempelvis i en vektordatabas som FAISS. I samband med detta behöver även rättigheter och åtkomststyrning lösas. Ett företag kan behöva inloggning i appen för att kunna styra vilken användare som får åtkomst till vilken information. Vissa dokument, som generella policys, kan vara tillgängliga för alla, medan andra dokument, till exempel interna how-to-guider eller avdelningsspecifika rutiner, endast bör vara tillgängliga för vissa roller eller avdelningar.

En annan viktig utvecklingsdel gäller feedback och kvalitetssäkring. I nuvarande version kan användaren bara sätta tumme upp eller tumme ner på ett svar, men denna data används ännu inte vidare. På sikt skulle feedback kunna användas för att analysera om svaren är korrekta, relevanta eller om underlaget behöver uppdateras. Det öppnar också för frågor om dokumentägarskap och ansvar: vem ska få information om att ett dokument verkar ge svaga svar, vem ska kunna uppdatera det och hur ska systemet avgöra om användarfeedback är tillförlitlig nog för att användas i fortsatt förbättring?

Det finns också etiska och affärsmässiga utmaningar. Även med RAG och källreferenser finns risk för felaktiga eller ofullständiga svar, särskilt med mindre modeller. Därför är transparens viktig, till exempel genom att visa källchunks och dokumentreferenser. Samtidigt behöver ett företag ta ställning till informationssäkerhet, ansvar, behörigheter och hur mycket automatiskt lärande från användarfeedback som är lämpligt. Dessa frågor är centrala om lösningen ska gå från prototyp till kommersiell produkt.

I denna version har fokus legat på att få kärnflödet i en RAG-lösning att fungera korrekt. Jag har därför implementerat de centrala delarna: chunking, embeddings, semantisk sökning, svarsgenerering och evaluering. Persistent lagring i en vektordatabas som FAISS är ett naturligt nästa steg för en större produktionslösning, men i den här versionen prioriterades att först bygga och verifiera ett fungerande grundflöde.


---


## 🔮 Framtida förbättringar
- Persistent lagring av dokument och embeddings till disk.
- Stöd för fler filformat (ex HTML, MD, JSON).
- Direktinläsning från wiki-sidor och intranät via URL.
- Kategorisering och filtrering av dokument per avdelning.
- Byte från minnesbaserad lagring till persistent vector store med FAISS för bättre skalbarhet.
- Semantic chunking för bättre textuppdelning.
- Uppgradering till större lokal modell för minskad hallucineringsrisk i produktionsmiljö.
- Byte av embeddingsmodell till `intfloat/multilingual-e5-large` för bättre stöd för nordiska språk och engelska i flerspråkiga företagsmiljöer.
- Stöd för enkel anpassning av appens utseende per företag, till exempel logotyp, färger och visuell profil, så att lösningen känns igen av de anställda.
- Förbättrad validering av Gemini-läge i UI så att Gemini inte kan väljas eller användas när `GEMINI_API_KEY` saknas.
- Vidare utvärdering av chunking-strategi för olika dokumenttyper, till exempel tabeller, policydokument och längre handböcker, för att hitta bästa balans mellan svarskvalitet, kostnad och svarstid.

**Användarhantering och behörighet:**
- Inloggning för att komma åt appen.
- Single Sign-On (SSO) med Active Directory-koppling för företag som redan använder AD / Office 365.
- Behörighetsstyrning — inte alla användare ska kunna ladda upp, ersätta eller uppdatera dokument.
- Dokumentägare — tagga varje uppladdning med vilken inloggad användare som laddade upp dokumentet.
- Möjlighet att ersätta eller uppdatera befintliga dokument, med kontroll över vem som får göra det.
- Dokumentbehörighet per användare eller roll — vissa dokument ska bara vara tillgängliga för vissa avdelningar.

**Uppföljning, feedback och förvaltning:**
- Anonym sökhistorik för att förstå hur systemet används — vilka frågor ställs och om användarna får hjälp, utan att lagra identitet i analysläge.
- Permanent lagring av feedback per svar i stället för enbart under aktiv session.
- Möjlighet för användare att inte bara sätta 👍 / 👎, utan också ange kommentar och typ av problem, till exempel:
  - felaktigt svar
  - gammalt dokument
  - fel dokument hittades
  - saknad information
  - dokument behöver uppdateras
- Möjlighet att koppla feedback till vilka källor och dokument som användes i svaret, för att enklare kunna analysera orsaken till fel.
- Möjlighet att koppla feedback till användare eller inloggad identitet när uppföljning krävs.
- Stöd för att koppla feedback till dokumentägare och skicka notifiering när ett dokument verkar vara inaktuellt, felaktigt eller behöver uppdateras.
- Möjlighet att skilja mellan anonym kvalitetsfeedback för modellförbättring och identifierbar felrapportering för dokumentförvaltning.
- Möjlighet att använda insamlad feedback för att förbättra retrieval, chunking, dokumentkvalitet och framtida modellval.


---


## 🔒 Datasekretess
Datasekretess beror på vilken LLM-provider som används:
- **Ollama (lokal modell):** all data stannar på den egna maskinen.
- **Gemini (online-modell):** fråga och relevant kontext skickas till Googles API.

För verksamheter med höga sekretesskrav rekommenderas lokal körning via Ollama.


---


## 📚 Tekniker och bibliotek
| Komponent | Teknologi |
|-----------|-----------|
| Gränssnitt | Streamlit |
| Embeddings | KBLab/sentence-bert-swedish-cased |
| Lokal LLM | Ollama (llama3.1:8b, llama3.2:3b m.fl.) |
| Online LLM | Google Gemini via google-genai |
| PDF-inläsning | pypdf |
| DOCX-inläsning | python-docx |
| PPTX-inläsning | python-pptx |
| XLSX-inläsning | openpyxl |
| Vektorsökning | NumPy (cosine similarity) |

---


## 👨‍💻 Utvecklat av

Josefin Carlson — BI25, Machine Learning / AI
