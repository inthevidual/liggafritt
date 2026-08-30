# Ligga Fritt

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

## De två lägena, och varför de finns

ONNX-graferna har **statiska** indatamått. Upplösningen går alltså inte att
skruva ner i efterhand — den är inbakad i exporten. Det tvingar fram två
checkpoints i stället för en:

| Läge | Modell | Upplösning | fp16 / fp32 | Krav |
|---|---|---|---|---|
| Standard | [`studioludens/birefnet-lite-512`](https://huggingface.co/studioludens/birefnet-lite-512) | 512×512 | 94 / 183 MB | inget |
| Hög detalj | [`onnx-community/BiRefNet_lite-ONNX`](https://huggingface.co/onnx-community/BiRefNet_lite-ONNX) | 1024×1024 | 109 / 214 MB | WebGPU |

Båda är MIT och härstammar från
[`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet).
512-varianten är en omexport gjord för just webbläsarkörning; enligt dess
modellkort matchar utdata PyTorch-referensen exakt.

**Uppmätt här:** 1024-exporten dör på WASM med `std::bad_alloc` — dess
aktiveringar spränger WASM-heapen. 512-exporten blir klar på ungefär fem
sekunder på samma maskin. Därför är Standard förval, och Hög detalj erbjuds bara
där WebGPU kan bära den. Knappen är avstängd med en förklaring i övriga fall,
hellre än att låta någon välja något som inte kan fungera.

Den porträtttränade checkpointen
([`BiRefNet-portrait`](https://huggingface.co/onnx-community/BiRefNet-portrait-ONNX))
är bättre än båda på testflyende hår, men är 467 MB i fp16 mot 94 MB här. Det är
inte ett rimligt förval för ett verktyg man öppnar en gång.

### Licens, medvetet vald

Allt ovan är MIT och därmed användbart kommersiellt. BRIA:s RMBG-1.4 och
RMBG-2.0 är marginellt bättre på svårt hår men ligger under CC BY-NC 4.0 och får
alltså **inte** användas i kommersiellt eller redaktionellt arbete utan köpt
licens. De är därför inte med.

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

`styles/app.css` är det som är specifikt för Ligga Fritt.

## Märket

`brand/mark-placeholder.svg` är en **platshållare** — en urklippt byst mot
transparensrutmönster, ritad i CP-Tjänsts palett och avsiktligt enkel så att den
läses som ett provisorium. Detsamma gäller `brand/og.png`.

Byt båda när den riktiga formgivningen kommer:

1. Lägg den riktiga SVG:n som `brand/mark.svg`.
2. Peka om `<img class="brand-mark">` i `index.htm` och ikongenereringen.
3. `node brand/tools/build-og.mjs` för delningsbilden.
4. `node brand/tools/stamp-assets.mjs` — annars serverar webbläsare den gamla.

## Cache busting

`index.htm` refererar lokala filer som `sökväg?v=<8 tecken innehållshash>`. Kör

```
node brand/tools/stamp-assets.mjs
```

efter varje ändring i `styles/`, `script.js`, `vendor/` eller `brand/`. Det
mönstret finns för att en versionsstämpel som inte ändras med innehållet gjorde
att webbläsare serverade gammal CSS i systerprojektet.

## Kvar att göra

- Riktigt märke och OG-bild.
- Domän och `CNAME` — sidan antar `liggafritt.cptjanst.se` i OG-taggarna.
- Ingen kantjustering ännu (tröskel, ludd, urkantning). BiRefNets matta är mjuk
  och brukar duga rakt av; lägg bara till reglage om verkligt bruk kräver det.
