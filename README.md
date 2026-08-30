# Ligga fritt

Frilägger porträtt till PNG med genomskinlig bakgrund. Allt sker i webbläsaren —
bilderna laddas aldrig upp någonstans.

Systerprojekt till [The Banfinator](../thebanfinator). Statisk sajt, ingen
byggkedja, samma designsystem.

---

## Så fungerar det

Matten kommer från **BiRefNet** (MIT), körd genom `transformers.js` på WebGPU där
det finns och WASM annars. Filen avkodas lokalt, modellen kör lokalt, PNG:en
skrivs lokalt.

Till skillnad från The Banfinator är sidan alltså *inte* fri från
tredjepartstrafik: biblioteket hämtas från jsDelivr och modellvikterna från
Hugging Face. Båda hämtas en gång och ligger sedan i webbläsarens cache.

## Modell och beräkning

| | |
|---|---|
| Modell | [`studioludens/birefnet-lite-512`](https://huggingface.co/studioludens/birefnet-lite-512) |
| Upplösning | 512×512, sedan uppskalad matta |
| Nedladdning | 94 MB (fp16, WebGPU) / 183 MB (fp32, WASM) |
| Licens | MIT |

En omexport av [`ZhengPeng7/BiRefNet_lite`](https://huggingface.co/ZhengPeng7/BiRefNet_lite)
gjord för webbläsarkörning; enligt modellkortet matchar utdata PyTorch-referensen
exakt.

### Varför bara en modell

BiRefNets ONNX-grafer har **statiska** indatamått, så upplösningen är ett val av
checkpoint och inte en inställning. 1024-exporten har ingen fungerande
konfiguration i en webbläsare:

- **WASM:** `std::bad_alloc` — aktiveringarna spränger WASM-heapen.
- **WebGPU:** shadern faller på gränsen nedan.

Så 512 är det som finns, och det tar ungefär fem sekunder.

Den porträtttränade
[`BiRefNet-portrait`](https://huggingface.co/onnx-community/BiRefNet-portrait-ONNX)
är bättre på flyende hår men är 467 MB i fp16 mot 94 MB här — inte rimligt för
ett verktyg man öppnar en gång.

### Gränsen som avgör WebGPU

ONNX Runtimes WebGPU-backend binder ungefär en lagringsbuffert per körtidsindata
plus en per utdata, och avbryter när en enskild kernel överskrider enhetens
`maxStorageBuffersPerShaderStage`:

```
numbers_storage_buffers_ <= limits_.maxStorageBuffersPerShaderStage
Too many storage buffers in shader. Current: 11, Max is 10
```

Bred `Concat` är den vanliga orsaken — varje indata är sin egen buffert. Kör
`tools/onnx-buffers.py` för att mäta:

| Modell | Värsta kernel |
|---|---|
| `studioludens/birefnet-lite-512` | **7** buffertar |
| `onnx-community/BiRefNet_lite` (1024) | **1025** — en `Concat` med 1024 indata |

Det är alltså 1024-exporten som havererar, inte arkitekturen. Den som körs här
behöver 7, under de 8 som WebGPU-specifikationen garanterar, så i praktiken
klarar alla enheter den. Kontrollen finns ändå kvar: den är gratis, den läses av
adaptern innan något laddas ner, och den fångar dagen siffran rör på sig.

Kontrollen täcker bara den gräns vi känner till — en drivrutin kan fela på sätt
som inget annonserar. Därför faller ett WebGPU-fel ändå tillbaka på WASM, minns
det till nästa besök och säger till. WASM är golvet.

### Om 1024 någon gång ska köras

Det skulle kräva grafkirurgi: dela upp de breda `Concat`-noderna i kedjor av
smala. Numeriskt identiskt, en engångsåtgärd på exporten. Men modellen får
dessutom slut på minne på WASM, så den skulle bara fungera med WebGPU — och
`/decoder/Concat` med 1024 indata är en större sak än den låter. Inte gjort,
och inte nödvändigt för det här verktyget.

## Byline

Verktygets huvudsakliga användning är porträttet i ett artikelhuvud, så
resultatet går att placera: dra i bilden, rulla för att zooma, eller använd
reglaget. **Stora förhandsvisningen och artikelhuvudet delar samma utsnitt** —
drar man i den ena flyttas den andra. Den stora visar var utsnittet ligger med
en ram och dämpad omgivning; artikelhuvudet visar hur det faktiskt ser ut.

### Mätt, inte gissat

Layouten är avläst från den riktiga artikeln med Playwright, inte antagen:

| | |
|---|---|
| Rubrik | Sueca Hd i **light**, 42/46 px, ingen negativ knipning |
| Skribentnamn | samma rubrik, som kolonprefix i `#C3C2C1` — **samma grå i båda lägena** |
| Taggrad | SvD Ester Blenda 700, sektionen i aktionsblått |
| Ljust läge | botten `#FFFFFF`, rubrik `#1D1D1B`, tagg `#0D4C80`, linje `#F2EFEE` |
| Mörkt läge | botten `#0B2337`, rubrik `#F7F6F5`, tagg `#42C0F0` |
| Porträtt | kvadratisk ruta, `overflow: hidden`, bottenlinjerad mot strecket |

Mörkt läge använder alltså `#0B2337` — exakt samma marin som `--blue-700` i
designsystemet. Och artikeln laddar precis de woff2-filer som ligger i `fonts/`,
vilket bekräftar att designsystemets typsnitt är produktionstypsnitten.

Både ljust och mörkt går att växla mellan, liksom desktop och mobil, eftersom
rubriken bryts olika och porträttet är mindre i den smala spalten.

Startläget fyller ramens bredd med motivet och ställer det på nederkanten, så
figuren går ut genom strecket i stället för att sväva. Bylinen sparas som PNG
med genomskinlig bakgrund, beskuren precis som i förhandsvisningen.

Förhandsvisningen är avsiktligt omärkt. Den visar placering och är inte en
återgivning av någon tidnings sidhuvud.

## Filformat

JPG, PNG och AVIF avkodas av webbläsaren själv. **TIFF har ingen inbyggd
avkodare i någon webbläsare**, så den går genom `vendor/utif.js`
([UTIF](https://github.com/photopea/UTIF.js), MIT). Ovanliga TIFF-varianter kan
misslyckas — då syns ett tydligt fel, aldrig ett tomt resultat.

## Att köra lokalt

```
python3 -m http.server 8788
```

Sidan måste serveras över http, inte öppnas som `file://` — modulimporter och
typsnitt faller annars på CORS.

## Designsystem

`styles/brand.css` är delat med The Banfinator och byte-identiskt med kopian
där. Det implementerar `designsystem/brand-book.md` §2–§5: råskalor, semantiska
roller och de tre temablocken. Rör det bara om det ska ändras i båda projekten,
och kopiera i så fall över filen i stället för att redigera en av dem.

`styles/app.css` är det som är specifikt för Ligga fritt.

## Märket

`brand/liggafritt-source.svg` är Illustrator-exporten. Allt annat under `brand/`
byggs från den:

```
node brand/tools/build-mark.mjs    # mark.svg, mark.webp, ikoner
node brand/tools/build-og.mjs      # og.png
node brand/tools/stamp-assets.mjs  # nya innehållshashar i index.htm
```

Bygget optimerar vektorn (188 → 123 kB) och slår ihop de färgpar som ligger
under den perceptuella tröskeln — tre par här, ΔE 0,84 till 1,37, palett 15 → 12
färger. Avsiktliga tonsteg ligger långt över ΔE 2 och lämnas orörda.

**Sidan laddar inte SVG:n.** Märket serveras som `mark.webp` (17 kB) eftersom
det är ett tecknat porträtt med ~500 paths: vektorn förblir stor hur hårt den än
pressas, medan en 160px webp täcker 3× DPR vid sidhuvudets 45 px. Vid 16–48 px
vinner en favicon ingenting på att vara vektor. `mark.svg` är den redigerbara
mastern.

Märket är **beskuret till ansiktet**. Uppmätt ur originalet — figuren renderad
utan halvtonen, silhuettens bredd läst rad för rad — sitter hjässan på y 92,
skäggets bredaste punkt kring y 420, halsen som smalast vid y 620 och axlarna
börjar flälla ut vid y 668. Huvudet upptar alltså y 92..620 och x 456..902, och
beskärningen `369, 52, 620×620` ramar in det med lite luft över och en aning
krage under.

Det beskär halon, vilket en helbild inte hade gjort. Det är avvägningen: vid
40 px i sidhuvudet blir ansiktet omkring femton pixlar högt i en helbild, och en
bestämd ansiktsbeskärning läses som avsiktlig där en nästan hel halo läses som
ett misstag.

OG-bilden använder däremot **hela figuren**, förankrad i nedre högra hörnet och
något överstor så att armar och överkropp går ut genom kortets egna kanter. Låg
den fritt med luft under såg den nedre beskärningen ut som ett fel i stället för
som ett utfall.

## Cache busting

`index.htm` refererar lokala filer som `sökväg?v=<8 tecken innehållshash>`. Kör

```
node brand/tools/stamp-assets.mjs
```

efter varje ändring i `styles/`, `script.js`, `vendor/` eller `brand/`. Det
mönstret finns för att en versionsstämpel som inte ändras med innehållet gjorde
att webbläsare serverade gammal CSS i systerprojektet.

## Kvar att göra

- Domän och `CNAME` — sidan antar `liggafritt.cptjanst.se` i OG-taggarna.
- Ingen kantjustering ännu (tröskel, ludd, urkantning). BiRefNets matta är mjuk
  och brukar duga rakt av; lägg bara till reglage om verkligt bruk kräver det.
