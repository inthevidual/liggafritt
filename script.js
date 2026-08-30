/* =============================================================================
   Ligga fritt — portrait background removal, entirely in the browser.

   The matte comes from BiRefNet (MIT), run through transformers.js on WebGPU
   where available and WASM otherwise. Nothing is uploaded: the file is decoded
   locally, the model runs locally, and the PNG is written locally.

   The only third-party traffic is the library from jsDelivr and the model
   weights from Hugging Face, both fetched once and then served from the
   browser's cache.
   ========================================================================== */

import {
    AutoModel,
    AutoProcessor,
    RawImage,
    env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

// Weights come from the Hub, never from this origin.
env.allowLocalModels = false;

/**
 * One model, and the reasons for that are measured rather than assumed.
 *
 * BiRefNet's ONNX graphs carry *static* input shapes, so resolution cannot be
 * turned down at runtime — it has to be chosen as a checkpoint. The 1024 export
 * has no working configuration in a browser: on WASM it dies with
 * std::bad_alloc, its activations exceeding the heap, and on WebGPU it hits the
 * shader limit below. So 512 it is, which finishes in about five seconds.
 *
 * The portrait-tuned checkpoint is better on flyaway hair but is 467 MB in fp16
 * against 94 MB here, which is not a reasonable ask for a tool opened once.
 */
const MODEL = {
    id: 'studioludens/birefnet-lite-512',
    name: 'BiRefNet lite, 512',
    licence: 'MIT',
    resolution: 512,
    size: { fp16: 94, fp32: 183 },
};

/**
 * ONNX Runtime's WebGPU backend binds about one storage buffer per runtime
 * input plus one per output, and aborts when a single kernel exceeds the
 * device's maxStorageBuffersPerShaderStage:
 *
 *   numbers_storage_buffers_ <= limits_.maxStorageBuffersPerShaderStage
 *   Too many storage buffers in shader. Current: 11, Max is 10
 *
 * Wide Concat is the usual cause — every input is its own buffer. Measured with
 * tools/onnx-buffers.py, this model's worst kernel binds 7; the 1024 export has
 * a Concat with 1024 inputs, which is why it cannot run here at all.
 *
 * 7 is under the 8 the WebGPU spec guarantees, so in practice every device
 * qualifies. The check stays anyway: it is cheap, it is read off the adapter
 * before anything is downloaded, and it will catch the day this number moves.
 */
const STORAGE_BUFFERS_NEEDED = 7;

const WASM_ONLY_KEY = 'lf.wasmOnly';

// The uploaded square, in pixels. SvD requests it at w=200 and renders it at
// 200 in the article header and 180 on the author page, so 800 leaves room for
// retina without making the file silly.
const SQUARE = 800;

const ACCEPTED = /\.(jpe?g|png|tiff?|avif)$/i;
const MAX_PIXELS = 40e6; // ~40 MP, above which decoding is the bottleneck, not the model

class LiggaFritt {
    constructor() {
        this.dropZone = document.getElementById('dropZone');
        this.fileInput = document.getElementById('fileInput');
        this.statusCard = document.getElementById('statusCard');
        this.statusLabel = document.getElementById('statusLabel');
        this.statusDetail = document.getElementById('statusDetail');
        this.progress = document.getElementById('progress');
        this.progressBar = document.getElementById('progressBar');
        this.resultCard = document.getElementById('resultCard');
        this.resultMeta = document.getElementById('resultMeta');
        this.canvasFrame = document.getElementById('canvasFrame');
        this.canvas = document.getElementById('previewCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.factModel = document.getElementById('factModel');
        this.factDevice = document.getElementById('factDevice');
        this.deviceNote = document.getElementById('deviceNote');
        this.bylineCard = document.getElementById('bylineCard');
        this.bylineZoom = document.getElementById('bylineZoom');
        // The upload itself: one square, drawn once per change and then blitted
        // into every preview. SvD serves the same file to the article header and
        // the author page, so there is one image to frame, not two.
        this.square = document.createElement('canvas');
        this.square.width = SQUARE;
        this.square.height = SQUARE;
        this.squareCtx = this.square.getContext('2d');
        this.views = [
            { frame: document.getElementById('artikelFrame'), canvas: document.getElementById('artikelCanvas') },
            { frame: document.getElementById('avatarFrame'), canvas: document.getElementById('avatarCanvas') },
        ].filter((v) => v.frame && v.canvas);

        this.source = null;   // { bitmap, width, height, name }
        this.cutout = null;   // ImageData with the matte applied
        this.view = 'cutout';
        // How the cut-out sits inside the byline frame. scale is a multiple of
        // the cover-fit baseline; offsets are in frame pixels.
        this.byline = { scale: 1, offsetX: 0, offsetY: 0 };
        this.drag = null;
        this.modelPromise = null;
        this.device = null;
        this.busy = false;

        // Set once WebGPU has proved unusable here, so the next visit does not
        // repeat a failure the user has already sat through.
        try { this.wasmOnly = localStorage.getItem(WASM_ONLY_KEY) === '1'; } catch { this.wasmOnly = false; }

        this.bindEvents();
        this.detectDevice();
    }

    /* ── Capability ─────────────────────────────────────────────────────── */

    /**
     * Pick the backend up front, and only claim WebGPU when the adapter can
     * actually carry this model's shader. An adapter existing is not enough:
     * the request can fail on a blocklisted driver, and even a healthy adapter
     * may publish a storage-buffer ceiling below what BiRefNet binds.
     */
    async detectDevice() {
        let adapter = null;
        if (navigator.gpu && !this.wasmOnly) {
            try { adapter = await navigator.gpu.requestAdapter(); } catch { adapter = null; }
        }

        const buffers = adapter?.limits?.maxStorageBuffersPerShaderStage ?? 0;
        const enough = buffers >= STORAGE_BUFFERS_NEEDED;
        this.device = adapter && enough ? 'webgpu' : 'wasm';

        const dtype = this.device === 'webgpu' ? 'fp16' : 'fp32';
        this.factModel.textContent = `${MODEL.name} (${MODEL.licence}) — ${MODEL.size[dtype]} MB`;

        if (this.device === 'webgpu') {
            this.factDevice.textContent = 'WebGPU — någon sekund per bild';
            this.deviceNote.textContent = '';
            return;
        }

        this.factDevice.textContent = 'WASM — några sekunder per bild';
        this.deviceNote.textContent =
            this.wasmOnly ? 'WebGPU gav fel här tidigare och används inte längre.'
            : !adapter ? 'Ingen WebGPU i den här webbläsaren.'
            : `Grafikkortet klarar ${buffers} lagringsbuffertar per shader, modellen behöver ${STORAGE_BUFFERS_NEEDED}.`;
    }

    /* ── Events ─────────────────────────────────────────────────────────── */

    bindEvents() {
        this.dropZone.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
            e.target.value = ''; // so picking the same file twice still fires
        });

        this.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropZone.classList.add('dragover');
        });
        this.dropZone.addEventListener('dragleave', () => this.dropZone.classList.remove('dragover'));
        this.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropZone.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.files);
        });

        // Dropping anywhere on the page is what people try first.
        document.addEventListener('dragover', (e) => e.preventDefault());
        document.addEventListener('drop', (e) => {
            if (this.dropZone.contains(e.target)) return;
            e.preventDefault();
            this.handleFiles(e.dataTransfer.files);
        });

        document.querySelectorAll('[data-view]').forEach((btn) => {
            btn.addEventListener('click', () => this.setView(btn.dataset.view));
        });

        document.querySelectorAll('[data-ground]').forEach((btn) => {
            if (btn.tagName !== 'BUTTON') return;
            btn.addEventListener('click', () => this.setGround(btn.dataset.ground));
        });

        this.bindByline();

        // Preview boxes change size with the window; the square does not, but
        // the backing stores blitted into them do.
        window.addEventListener('resize', () => {
            if (this.cutout) this.paintFraming();
        });

        this.downloadBtn.addEventListener('click', () => this.download());
        this.resetBtn.addEventListener('click', () => this.reset());
    }

    /* ── Status ─────────────────────────────────────────────────────────────
       §9: a failure must read as a failure. Never leave the last good message
       standing when something has gone wrong, and never show a bare zero.
       ------------------------------------------------------------------- */

    setStatus(label, detail = '', tone = 'working') {
        this.statusCard.hidden = false;
        this.statusLabel.textContent = label;
        this.statusDetail.textContent = detail;
        this.statusCard.dataset.tone = tone;
    }

    setProgress(fraction) {
        if (fraction === null) {
            this.progress.hidden = true;
            return;
        }
        this.progress.hidden = false;
        this.progressBar.style.width = `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
    }

    fail(label, detail) {
        this.setStatus(label, detail, 'critical');
        this.setProgress(null);
        this.busy = false;
        this.dropZone.classList.remove('is-busy');
    }

    /* ── Input ──────────────────────────────────────────────────────────── */

    async handleFiles(files) {
        const file = files?.[0];
        if (!file || this.busy) return;

        if (!ACCEPTED.test(file.name) && !/^image\//.test(file.type)) {
            this.fail('Filformatet stöds inte', `${file.name} är varken JPG, PNG, TIFF eller AVIF.`);
            return;
        }

        this.busy = true;
        this.dropZone.classList.add('is-busy');
        this.setProgress(null);

        try {
            this.setStatus('Läser in bilden', file.name);
            const bitmap = await this.decode(file);

            if (bitmap.width * bitmap.height > MAX_PIXELS) {
                bitmap.close?.();
                this.fail('Bilden är för stor',
                    `${bitmap.width}×${bitmap.height} px. Skala ner till under ${MAX_PIXELS / 1e6} megapixel först.`);
                return;
            }

            this.source?.bitmap?.close?.();
            this.source = { bitmap, width: bitmap.width, height: bitmap.height, name: file.name };
            this.dropZone.classList.add('loaded');
            document.getElementById('dropName').textContent =
                `${file.name} — ${bitmap.width}×${bitmap.height} px`;

            await this.run();
        } catch (err) {
            console.error(err);
            this.fail('Kunde inte läsa bilden', String(err?.message || err));
        }
    }

    /**
     * TIFF has no native decoder in any browser, so it goes through UTIF;
     * everything else goes through the browser's own image pipeline.
     */
    async decode(file) {
        const isTiff = /\.tiff?$/i.test(file.name) || file.type === 'image/tiff';
        if (!isTiff) {
            try {
                return await createImageBitmap(file);
            } catch (err) {
                throw new Error(`webbläsaren kunde inte avkoda ${file.type || 'filen'}`);
            }
        }

        if (typeof UTIF === 'undefined') throw new Error('TIFF-avkodaren kunde inte laddas');

        const buffer = await file.arrayBuffer();
        const pages = UTIF.decode(buffer);
        if (!pages.length) throw new Error('TIFF-filen innehåller inga bilder');

        UTIF.decodeImage(buffer, pages[0]);
        const rgba = UTIF.toRGBA8(pages[0]);
        const { width, height } = pages[0];
        if (!width || !height) throw new Error('TIFF-filen saknar giltiga mått');

        const data = new ImageData(new Uint8ClampedArray(rgba.buffer), width, height);
        return await createImageBitmap(data);
    }

    /* ── Model ──────────────────────────────────────────────────────────── */

    loadModel() {
        if (this.modelPromise) return this.modelPromise;

        // Per-file download fractions, combined into one bar — transformers.js
        // reports each shard separately and a bar that restarts looks broken.
        const files = new Map();

        const onProgress = (p) => {
            if (p.status === 'progress' && p.total) {
                files.set(p.file, { loaded: p.loaded, total: p.total });
                let loaded = 0;
                let total = 0;
                files.forEach((f) => { loaded += f.loaded; total += f.total; });
                this.setProgress(loaded / total);
                this.setStatus('Hämtar modellen',
                    `${this.mb(loaded)} av ${this.mb(total)} — sparas i webbläsarens cache, sker bara första gången`);
            }
        };

        // fp16 halves the download but is only reliable on WebGPU; the WASM
        // backend gets fp32 rather than a matte full of NaNs.
        const dtype = this.device === 'webgpu' ? 'fp16' : 'fp32';

        this.modelPromise = (async () => {
            const [model, processor] = await Promise.all([
                AutoModel.from_pretrained(MODEL.id, { dtype, device: this.device, progress_callback: onProgress }),
                AutoProcessor.from_pretrained(MODEL.id, { progress_callback: onProgress }),
            ]);
            return { model, processor };
        })();

        // Let a failed load be retried rather than cached as broken.
        this.modelPromise.catch(() => { this.modelPromise = null; });
        return this.modelPromise;
    }

    mb(bytes) {
        return `${(bytes / 1048576).toFixed(0)} MB`;
    }

    /* ── Inference ──────────────────────────────────────────────────────── */

    async run() {
        if (!this.source) return;
        const { bitmap, width, height } = this.source;

        let model;
        let processor;
        try {
            await this.detectDevice();
            ({ model, processor } = await this.loadModel());
        } catch (err) {
            console.error(err);
            if (await this.retryOnWasm('Modellen kunde inte laddas på grafikkortet')) return;
            this.fail('Modellen kunde inte laddas',
                'Kontrollera nätverket och försök igen. Inget har skickats någonstans.');
            return;
        }

        this.setProgress(null);
        this.setStatus('Frilägger', `${width}×${height} px — modellen arbetar i ${MODEL.resolution}×${MODEL.resolution}`);

        try {
            const started = performance.now();

            const rgba = this.toImageData(bitmap, width, height);
            const image = new RawImage(new Uint8ClampedArray(rgba.data), width, height, 4).rgb();

            const { pixel_values } = await processor(image);
            const { output_image } = await model({ input_image: pixel_values });

            const mask = await RawImage.fromTensor(
                output_image[0].sigmoid().mul(255).to('uint8')
            ).resize(width, height);

            this.cutout = this.applyMatte(rgba, mask);
            // drawImage scales, putImageData does not — the byline frame needs
            // the former, so keep a bitmap alongside the pixels.
            this.cutoutBitmap?.close?.();
            this.cutoutBitmap = await createImageBitmap(this.cutout);
            this.bounds = null;
            const seconds = ((performance.now() - started) / 1000).toFixed(1);

            this.showResult(seconds);
        } catch (err) {
            console.error(err);
            if (await this.retryOnWasm('Beräkningen misslyckades på grafikkortet')) return;
            this.fail('Friläggningen misslyckades', String(err?.message || err));
        }
    }

    /**
     * The adapter check above should keep us off a GPU that cannot run this
     * model, but it only covers the one limit we know about — a driver can
     * still fail in ways nothing advertises. So a WebGPU failure drops to WASM,
     * remembers that for next time, and retries, rather than presenting a dead
     * end. WASM is the floor: if that fails there is nowhere further to go.
     */
    async retryOnWasm(reason) {
        if (this.device !== 'webgpu') return false;

        this.wasmOnly = true;
        this.modelPromise = null;
        try { localStorage.setItem(WASM_ONLY_KEY, '1'); } catch { /* private mode */ }

        await this.detectDevice();
        this.setStatus('Byter till WASM', `${reason} — kör om utan grafikacceleration.`);
        await this.run();
        return true;
    }

    toImageData(bitmap, width, height) {
        const canvas = new OffscreenCanvas(width, height);
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(bitmap, 0, 0);
        return ctx.getImageData(0, 0, width, height);
    }

    /** Full-resolution RGB from the source, alpha from the model's matte. */
    applyMatte(rgba, mask) {
        const out = new ImageData(new Uint8ClampedArray(rgba.data), rgba.width, rgba.height);
        const alpha = mask.data;
        const step = mask.channels; // the mask comes back single-channel, but do not assume
        for (let i = 0, a = 0; i < out.data.length; i += 4, a += step) {
            out.data[i + 3] = alpha[a];
        }
        return out;
    }

    /* ── Byline framing ─────────────────────────────────────────────────────
       The cut-out sits free on the article's dark header block, right-aligned
       and bled through the rule that closes it. Nothing about that placement is
       automatic, so the framing is direct: drag to move, wheel to scale.
       ------------------------------------------------------------------- */

    bindByline() {
        if (!this.views.length) return;

        // Every preview is a window onto the same square, so each one drags it;
        // they differ only in how many square-pixels one on-screen pixel is.
        this.views.forEach((v) => {
            const ratio = () => SQUARE / v.frame.getBoundingClientRect().width;
            this.bindFraming(v.frame, ratio, (e) => {
                const r = v.frame.getBoundingClientRect();
                const k = ratio();
                return { x: (e.clientX - r.left) * k, y: (e.clientY - r.top) * k };
            });
        });

        this.bylineZoom.addEventListener('input', (e) => {
            this.setBylineScale(parseFloat(e.target.value));
        });

        document.querySelectorAll('[data-viewport]').forEach((btn) => {
            if (btn.tagName !== 'BUTTON') return;
            btn.addEventListener('click', () => this.setViewport(btn.dataset.viewport));
        });

        document.querySelectorAll('[data-scheme]').forEach((btn) => {
            if (btn.tagName !== 'BUTTON') return;
            btn.addEventListener('click', () => this.setScheme(btn.dataset.scheme));
        });

        // The big preview is the same framing at working size: dragging either
        // moves the same crop, so the article header updates as you work.
        this.bindFraming(this.canvas, () => this.previewScale(), (e) => this.anchorInPreview(e));

        const author = document.getElementById('authorInput');
        const headline = document.getElementById('headlineInput');
        const syncText = () => {
            const name = author.value.trim();
            // The name is a colon prefix inside the headline, so the colon comes
            // from us — typing one should not produce two.
            const clean = name.replace(/:\s*$/, '');
            document.getElementById('previewAuthor').textContent = clean ? clean + ':' : '';
            document.getElementById('previewName').textContent = clean;
            document.getElementById('previewHeadline').textContent = headline.value;
        };
        author.addEventListener('input', syncText);
        headline.addEventListener('input', syncText);

        document.getElementById('bylineResetBtn').addEventListener('click', () => this.fitByline());
        document.getElementById('bylineDownloadBtn').addEventListener('click', () => this.downloadByline());
    }

    /**
     * Make a surface frame the cut-out. Both the byline frame and the big
     * preview drive the same transform, so a drag in either lands in both; they
     * differ only in how many frame-pixels one surface-pixel is worth, which is
     * what `ratio` returns.
     */
    bindFraming(el, ratio, anchor) {
        el.addEventListener('pointerdown', (e) => {
            if (!this.cutout) return;
            el.setPointerCapture(e.pointerId);
            this.drag = { x: e.clientX, y: e.clientY, ox: this.byline.offsetX, oy: this.byline.offsetY };
        });

        el.addEventListener('pointermove', (e) => {
            if (!this.drag) return;
            const k = ratio();
            this.byline.offsetX = this.drag.ox + (e.clientX - this.drag.x) * k;
            this.byline.offsetY = this.drag.oy + (e.clientY - this.drag.y) * k;
            this.paintFraming();
        });

        const end = (e) => {
            if (!this.drag) return;
            this.drag = null;
            el.releasePointerCapture?.(e.pointerId);
        };
        el.addEventListener('pointerup', end);
        el.addEventListener('pointercancel', end);

        el.addEventListener('wheel', (e) => {
            if (!this.cutout) return;
            e.preventDefault();
            this.setBylineScale(this.byline.scale * Math.exp(-e.deltaY * 0.0015), anchor(e));
        }, { passive: false });
    }

    /**
     * One big-preview pixel in byline-frame pixels. The big preview shows the
     * whole cut-out letterboxed; the byline frame shows a window onto it at its
     * own scale, so the two differ by the ratio of those scales.
     */
    previewScale() {
        const fit = this.previewFit();
        const k = (this.byline.base || 1) * this.byline.scale;
        const css = this.canvas.getBoundingClientRect().width / this.canvas.width;
        return k / (fit.scale * css);
    }

    /** A pointer event over the big preview, mapped into frame pixels. */
    anchorInPreview(e) {
        const r = this.canvas.getBoundingClientRect();
        const fit = this.previewFit();
        const css = r.width / this.canvas.width;
        const k = (this.byline.base || 1) * this.byline.scale;
        const ix = ((e.clientX - r.left) / css - fit.x) / fit.scale;
        const iy = ((e.clientY - r.top) / css - fit.y) / fit.scale;
        return { x: ix * k + this.byline.offsetX, y: iy * k + this.byline.offsetY };
    }

    /** Letterbox geometry for the cut-out inside the big preview canvas. */
    previewFit() {
        const c = this.canvas;
        const scale = Math.min(c.width / this.cutout.width, c.height / this.cutout.height);
        return {
            scale,
            x: (c.width - this.cutout.width * scale) / 2,
            y: (c.height - this.cutout.height * scale) / 2,
        };
    }

    paintFraming() {
        this.paintByline();
        this.paint();
    }

    setScheme(which) {
        document.getElementById('artikelPreview').dataset.scheme = which;
        document.getElementById('forfattarePreview').dataset.scheme = which;
        document.querySelectorAll('[data-scheme]').forEach((btn) => {
            if (btn.tagName !== 'BUTTON') return;
            const active = btn.dataset.scheme === which;
            btn.classList.toggle('is-active', active);
            btn.setAttribute('aria-pressed', String(active));
        });
    }

    setViewport(which) {
        document.getElementById('artikelPreview').dataset.viewport = which;
        document.querySelectorAll('[data-viewport]').forEach((btn) => {
            if (btn.tagName !== 'BUTTON') return;
            const active = btn.dataset.viewport === which;
            btn.classList.toggle('is-active', active);
            btn.setAttribute('aria-pressed', String(active));
        });
        this.paintFraming();
    }

    setBylineScale(scale, anchor = null) {
        const next = Math.min(6, Math.max(1, scale));
        const prev = this.byline.scale;
        if (next === prev) return;

        // Zoom about a point so what is under it stays put — the cursor when
        // scrolling, the middle of the frame when using the slider. Scaling
        // about the origin instead just shoves the subject out of view.
        const at = anchor || { x: SQUARE / 2, y: SQUARE / 2 };
        const k = next / prev;
        this.byline.offsetX = at.x - (at.x - this.byline.offsetX) * k;
        this.byline.offsetY = at.y - (at.y - this.byline.offsetY) * k;
        this.byline.scale = next;
        this.bylineZoom.value = next;
        this.paintFraming();
    }


    /** The tight bounding box of everything the matte kept. */
    subjectBounds() {
        const { data, width, height } = this.cutout;
        let minX = width; let minY = height; let maxX = -1; let maxY = -1;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                if (data[(y * width + x) * 4 + 3] > 24) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }
        if (maxX < 0) return { x: 0, y: 0, width, height };
        return { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
    }

    /**
     * Default framing: fill the square's width with the subject and sit it on
     * the bottom edge. That reads correctly in both places the square is used —
     * whole, standing on the article's rule, and circle-cropped on the author
     * page, where a bottom-anchored figure fills the circle rather than floating
     * in it.
     */
    fitByline() {
        if (!this.cutout) return;
        const b = this.bounds || (this.bounds = this.subjectBounds());
        const base = SQUARE / b.width;

        this.byline.scale = 1;
        this.byline.base = base;
        this.byline.offsetX = -b.x * base + (SQUARE - b.width * base) / 2;
        this.byline.offsetY = SQUARE - (b.y + b.height) * base;

        this.bylineZoom.value = 1;
        this.paintFraming();
    }

    /** Draw the square once, then blit it into each preview. */
    paintByline() {
        if (!this.cutout || !this.squareCtx) return;
        const ctx = this.squareCtx;
        ctx.clearRect(0, 0, SQUARE, SQUARE);

        const k = (this.byline.base || 1) * this.byline.scale;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(this.cutoutBitmap, this.byline.offsetX, this.byline.offsetY,
            this.cutout.width * k, this.cutout.height * k);

        this.views.forEach(({ frame, canvas }) => {
            const size = Math.round(frame.getBoundingClientRect().width * 2) || 2;
            if (canvas.width !== size) { canvas.width = size; canvas.height = size; }
            const c = canvas.getContext('2d');
            c.clearRect(0, 0, size, size);
            c.imageSmoothingQuality = 'high';
            c.drawImage(this.square, 0, 0, size, size);
        });
    }

    async downloadByline() {
        if (!this.cutout) return;
        const blob = await new Promise((r) => this.square.toBlob(r, 'image/png'));
        if (!blob) {
            this.fail('Kunde inte skapa PNG', 'Webbläsaren nekade exporten.');
            return;
        }
        const author = document.getElementById('authorInput').value.trim() || 'byline';
        const slug = author.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '')
            .replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `${slug}_byline.png`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 10000);
    }

    /* ── Output ─────────────────────────────────────────────────────────── */

    showResult(seconds) {
        this.busy = false;
        this.dropZone.classList.remove('is-busy');
        this.statusCard.hidden = true;
        this.resultCard.hidden = false;
        this.bylineCard.hidden = false;
        this.downloadBtn.disabled = false;
        this.resultMeta.textContent =
            `${this.source.name} — ${this.source.width}×${this.source.height} px, frilagd på ${seconds} s`;
        this.setView('cutout');
        this.fitByline();
    }

    setView(view) {
        this.view = view;
        document.querySelectorAll('[data-view]').forEach((btn) => {
            const active = btn.dataset.view === view;
            btn.classList.toggle('is-active', active);
            btn.setAttribute('aria-pressed', String(active));
        });
        this.paint();
    }

    setGround(ground) {
        this.canvasFrame.dataset.ground = ground;
        document.querySelectorAll('.ground-swatch').forEach((btn) => {
            const active = btn.dataset.ground === ground;
            btn.classList.toggle('is-active', active);
            btn.setAttribute('aria-pressed', String(active));
        });
    }

    paint() {
        if (!this.source) return;
        const { width, height } = this.source;
        this.canvas.width = width;
        this.canvas.height = height;
        this.ctx.clearRect(0, 0, width, height);

        if (this.view === 'original' || !this.cutout) {
            this.ctx.drawImage(this.source.bitmap, 0, 0);
            return;
        }

        this.ctx.putImageData(this.cutout, 0, 0);
        this.drawCropOutline();
    }

    /**
     * Show where the byline frame currently sits, so the big preview is not
     * just a bigger picture but the same framing decision at working size.
     */
    drawCropOutline() {
        if (!this.byline.base) return;
        const ctx = this.ctx;
        const k = this.byline.base * this.byline.scale;

        // The frame, expressed in the cut-out's own pixels.
        const x = -this.byline.offsetX / k;
        const y = -this.byline.offsetY / k;
        const w = SQUARE / k;
        const h = SQUARE / k;

        const line = Math.max(2, this.canvas.width / 260);
        ctx.save();
        // Dim what falls outside it rather than drawing a bare rectangle — the
        // subject is often light, and a thin line on it is easy to lose.
        ctx.fillStyle = 'rgba(3, 20, 32, 0.34)';
        ctx.beginPath();
        ctx.rect(0, 0, this.canvas.width, this.canvas.height);
        ctx.rect(x, y, w, h);
        ctx.fill('evenodd');

        ctx.strokeStyle = getComputedStyle(document.documentElement)
            .getPropertyValue('--color-focus').trim() || '#0098DA';
        ctx.lineWidth = line;
        ctx.strokeRect(x, y, w, h);
        ctx.restore();
    }

    async download() {
        if (!this.cutout) return;

        // Compose off-screen so the download never depends on what the preview
        // happens to be showing.
        const canvas = document.createElement('canvas');
        canvas.width = this.cutout.width;
        canvas.height = this.cutout.height;
        canvas.getContext('2d').putImageData(this.cutout, 0, 0);

        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
        if (!blob) {
            this.fail('Kunde inte skapa PNG', 'Webbläsaren nekade exporten.');
            return;
        }

        const base = this.source.name.replace(/\.[^.]+$/, '');
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `${base}_frilagd.png`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 10000);
    }

    reset() {
        this.source?.bitmap?.close?.();
        this.cutoutBitmap?.close?.();
        this.source = null;
        this.cutout = null;
        this.cutoutBitmap = null;
        this.bounds = null;
        this.busy = false;
        this.dropZone.classList.remove('loaded', 'is-busy');
        document.getElementById('dropName').textContent = '';
        this.resultCard.hidden = true;
        this.bylineCard.hidden = true;
        this.statusCard.hidden = true;
        this.downloadBtn.disabled = true;
        this.fileInput.value = '';
    }
}

new LiggaFritt();
