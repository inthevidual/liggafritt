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
reglaget. Förhandsvisningen speglar hur huvudet är byggt —

- taggrad, sedan en `<h1>` vars **första span är skribentens namn som ett
  kolonprefix**, satt tillbaka i grått, följt av rubriken i vitt;
- porträttet högerställt i det mörka blocket och **beskuret av strecket som
  avslutar blocket**, inte en rund avatar;
- bildtext med fotobyline inline efter `Foto:`, aldrig som egen högerställd rad.

Desktop och mobil går att växla mellan, eftersom rubriken bryts olika och
porträttet är mindre i den smala spalten.

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

Ingen beskärning görs, till skillnad från Banfinatorns märke: den här
originalfilen ramar redan in sig själv. Uppmätt spänner halotonens gloria
x 102..1194, y 14..1080 i en kvadrat på 1254, alltså fri från alla kanter, och
bara axlarna går ut genom underkanten — vilket är rätt för en byst.

En sak att avgöra: märket visar hela bysten, så ansiktet blir omkring 15 px i
sidhuvudet. Banfinatorn beskärs till huvudet just därför. Säg till om samma
behandling ska göras här — det är en rad i `build-mark.mjs`.

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
