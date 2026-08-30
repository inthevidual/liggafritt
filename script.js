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
 * BiRefNet's compute shader binds 11 storage buffers. WebGPU devices advertise
 * a ceiling for that, and Chrome on Apple Silicon reports 10 — one short — so
 * the run aborts inside ONNX Runtime with
 *
 *   numbers_storage_buffers_ <= limits_.maxStorageBuffersPerShaderStage
 *   Too many storage buffers in shader. Current: 11, Max is 10
 *
 * The adapter publishes that number, so it is checkable before anything is
 * downloaded or run. Machines that cannot carry the shader quietly use WASM
 * instead of failing halfway through.
 */
const STORAGE_BUFFERS_NEEDED = 11;

const WASM_ONLY_KEY = 'lf.wasmOnly';

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

        this.source = null;   // { bitmap, width, height, name }
        this.cutout = null;   // ImageData with the matte applied
        this.view = 'cutout';
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

    /* ── Output ─────────────────────────────────────────────────────────── */

    showResult(seconds) {
        this.busy = false;
        this.dropZone.classList.remove('is-busy');
        this.statusCard.hidden = true;
        this.resultCard.hidden = false;
        this.downloadBtn.disabled = false;
        this.resultMeta.textContent =
            `${this.source.name} — ${this.source.width}×${this.source.height} px, frilagd på ${seconds} s`;
        this.setView('cutout');
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
        } else {
            this.ctx.putImageData(this.cutout, 0, 0);
        }
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
        this.source = null;
        this.cutout = null;
        this.busy = false;
        this.dropZone.classList.remove('loaded', 'is-busy');
        document.getElementById('dropName').textContent = '';
        this.resultCard.hidden = true;
        this.statusCard.hidden = true;
        this.downloadBtn.disabled = true;
        this.fileInput.value = '';
    }
}

new LiggaFritt();
