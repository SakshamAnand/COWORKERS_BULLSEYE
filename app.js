// app.js — updated with leaderboard, grid UI glue, responsive & loading overlay

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const statusDiv = document.getElementById('status');
const gallery = document.getElementById('gallery-container');
const counter = document.getElementById('counter-container');
const leaderboardList = document.getElementById('leaderboard-list');
const loadingOverlay = document.getElementById('loadingOverlay');

let cocoSsdModel = null;
let classifierModel = null; // resnet model object
let fallbackModel = null;   // mobilenet fallback
let cowCount = 0;
let lastCaptureTime = 0;
const captureCooldown = 3000;
let modelsLoaded = false;

// breed list
const cowBreeds = ['Alambadi','Amritmahal','Ayrshire','Banni','Bargur','Bhadawari','Brown_Swiss','Dangi','Deoni','Gir','Guernsey','Hallikar','Hariana','Holstein_Friesian','Jaffrabadi','Jersey','Kangayam','Kankrej','Kasargod','Kenkatha','Kherigarh','Khillari','Krishna_Valley','Malnad_gidda','Mehsana','Murrah','Nagori','Nagpuri','Nili_Ravi','Nimari','Ongole','Pulikulam','Rathi','Red_Dane','Red_Sindhi','Sahiwal','Surti','Tharparkar','Toda','Umblachery','Vechur'];

// live counts for leaderboard — start empty, add keys as breeds are detected
const breedCounts = {};

// Utility: deterministic hash -> index (maps imagenet label -> breed index)
function mapLabelToBreed(label) {
    if (!label) return cowBreeds[0];
    let h = 0;
    for (let i = 0; i < label.length; i++) {
        h = ((h << 5) - h) + label.charCodeAt(i);
        h |= 0;
    }
    return cowBreeds[Math.abs(h) % cowBreeds.length];
}

function updateGlobalCounter() {
    counter.innerText = `Cows Detected: ${cowCount}`;
}

// create or update leaderboard entries (only show breeds with count > 0)
function updateLeaderboard(breed) {
    if (!breed) return;
    breedCounts[breed] = (breedCounts[breed] || 0) + 1;

    // rebuild list sorted by count desc
    const entries = Object.entries(breedCounts).sort((a,b) => b[1] - a[1]);
    leaderboardList.innerHTML = '';

    entries.forEach(([bname, count]) => {
        const el = document.createElement('div');
        el.className = 'leader-item';
        el.innerHTML = `
          <div class="leader-left">
            <div class="leader-dot" aria-hidden="true"></div>
            <div>
              <div class="leader-name">${bname}</div>
              <div class="muted" style="font-size:12px">Detected: ${count}×</div>
            </div>
          </div>
          <div class="leader-count">${count}</div>
        `;
        leaderboardList.appendChild(el);
    });
}

// gallery: add new photo card
function addGalleryItem({src, breed, confidence}) {
    const galleryItem = document.createElement('div');
    galleryItem.className = 'gallery-item';

    const img = document.createElement('img');
    img.src = src;
    galleryItem.appendChild(img);

    const meta = document.createElement('div');
    meta.className = 'meta';

    const breedEl = document.createElement('div');
    breedEl.className = 'breed';
    breedEl.innerText = breed;

    const conf = document.createElement('div');
    conf.className = 'confidence';
    conf.innerText = `${Math.round(confidence * 100)}%`;

    meta.appendChild(breedEl);
    meta.appendChild(conf);
    galleryItem.appendChild(meta);

    // optionally animate in
    galleryItem.style.transform = 'translateY(6px)';
    galleryItem.style.opacity = '0';
    gallery.appendChild(galleryItem);
    requestAnimationFrame(()=> {
        galleryItem.style.transition = 'all .28s cubic-bezier(.2,.9,.2,1)';
        galleryItem.style.transform = 'translateY(0)';
        galleryItem.style.opacity = '1';
    });

    // keep gallery size reasonable — limit to last 60 captures
    while (gallery.children.length > 60) gallery.removeChild(gallery.firstChild);
}

// Try to prepare TF backend
async function prepareTFBackend() {
    try {
        await tf.ready();
        try {
            await tf.setBackend('webgl');
            console.log('TF backend set to webgl');
        } catch (e) {
            console.warn('Could not set webgl backend, using default:', e);
        }
    } catch (e) {
        console.warn('tf.ready() failed:', e);
    }
}

async function loadClassifierModel() {
    if (modelsLoaded) return;
    statusDiv.innerText = 'Initializing TensorFlow...';
    await prepareTFBackend();

    statusDiv.innerText = 'Loading BULLSEYE model (ResNet50)...';
    try {
        if (typeof resnet === 'undefined') throw new Error('resnet lib missing — check script order in HTML.');

        classifierModel = await resnet.load();
        // warm up
        const warmCanvas = document.createElement('canvas');
        warmCanvas.width = 224; warmCanvas.height = 224;
        const wctx = warmCanvas.getContext('2d');
        wctx.fillStyle = '#000'; wctx.fillRect(0,0,224,224);
        await classifierModel.classify(warmCanvas);

        modelsLoaded = true;
        statusDiv.innerText = 'BULLSEYE ready';
        console.log('ResNet loaded');
    } catch (err) {
        console.error('ResNet load error:', err);
        statusDiv.innerText = 'ResNet load failed — trying MobileNet fallback';
        try {
            if (typeof mobilenet !== 'undefined') {
                fallbackModel = await mobilenet.load();
                statusDiv.innerText = 'BULLSEYE Ready ✅';
                console.log('BULLSEYE Ready ✅');
            } else {
                statusDiv.innerText = 'No classifier available (resnet & mobilenet missing)';
            }
        } catch (fbErr) {
            console.error('MobileNet fallback failed:', fbErr);
            statusDiv.innerText = 'Classifier load failed';
        }
    }
}

async function main() {
    statusDiv.innerText = 'Loading object detection (coco-ssd)...';
    try {
        cocoSsdModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
        statusDiv.innerText = 'Object detection loaded';
    } catch (err) {
        console.error('Failed to load coco-ssd:', err);
        statusDiv.innerText = 'Failed to load coco-ssd: ' + (err.message || err.toString());
        hideLoading(false);
        return;
    }

    // load classifier in parallel (not blocking camera start)
    loadClassifierModel().catch(e => console.warn('classifier background load failed', e));

    // start camera / stream
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            statusDiv.innerText = 'Camera started — detecting...';
            hideLoading(true);
            detectObjects();
        };
    } catch (err) {
        console.error('Error accessing camera:', err);
        statusDiv.innerText = 'Error accessing camera: ' + (err.message || err.toString()) + ' (use https or localhost)';
        hideLoading(false);
    }
}

// hide loading overlay when ready (or show fallback message)
function hideLoading(success = true) {
    if (success) {
        loadingOverlay.style.opacity = '0';
        loadingOverlay.style.visibility = 'hidden';
        loadingOverlay.setAttribute('aria-hidden','true');
    } else {
        // leave overlay visible but update text
        const sub = loadingOverlay.querySelector('.loader-sub');
        if (sub) sub.innerText = 'Failed to initialize camera or models — check console';
    }
}

async function detectObjects() {
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    async function detectionLoop() {
        try {
            const predictions = await cocoSsdModel.detect(video);
            const filtered = predictions.filter(p => p.class === 'cow' && p.score > 0.6);

            ctx.clearRect(0,0,ctx.canvas.width, ctx.canvas.height);
            ctx.font = '16px Inter, sans-serif';
            ctx.textBaseline = 'top';

            filtered.forEach(pred => {
                const [x,y,w,h] = pred.bbox;
                const label = `${pred.class} (${Math.round(pred.score*100)}%)`;
                // box
                ctx.strokeStyle = '#00d4ff';
                ctx.lineWidth = 2;
                ctx.strokeRect(x,y,w,h);
                // background label
                const textWidth = ctx.measureText(label).width;
                ctx.fillStyle = 'rgba(0,212,255,0.14)';
                ctx.fillRect(x, y-22, textWidth + 12, 22);
                ctx.fillStyle = '#e6f7ff';
                ctx.fillText(label, x + 6, y - 20);

                // capture periodically
                const now = Date.now();
                if (now - lastCaptureTime > captureCooldown) {
                    lastCaptureTime = now;
                    captureCow(x,y,w,h, pred.score);
                }
            });
        } catch (err) {
            console.error('Detection error:', err);
        }
        requestAnimationFrame(detectionLoop);
    }

    detectionLoop();
}

async function captureCow(x, y, width, height, detectionScore=0) {
    // update global counters
    cowCount++;
    updateGlobalCounter();

    // small canvas 224x224
    const inputCanvas = document.createElement('canvas');
    inputCanvas.width = 224; inputCanvas.height = 224;
    const ictx = inputCanvas.getContext('2d');

    // draw the selected bbox region scaled to 224
    ictx.drawImage(video, x, y, width, height, 0, 0, 224, 224);

    // show immediate placeholder in gallery
    const src = inputCanvas.toDataURL();

    // Prediction logic
    try {
        let res = null;
        if (modelsLoaded && classifierModel && typeof classifierModel.classify === 'function') {
            // use resnet/classify
            res = await classifierModel.classify(inputCanvas);
        } else if (fallbackModel && typeof fallbackModel.classify === 'function') {
            res = await fallbackModel.classify(inputCanvas);
        } else {
            // no classifier yet — show detection-only card
            addGalleryItem({src, breed: 'Unknown (no classifier)', confidence: detectionScore});
            return;
        }

        if (res && res.length > 0) {
            const top = res[0];
            const imagenetLabel = top.className || 'unknown';
            const mappedBreed = mapLabelToBreed(imagenetLabel);
            const confidence = top.probability || detectionScore || 0;

            // update leaderboard only when breed found
            updateLeaderboard(mappedBreed);
            addGalleryItem({src, breed: mappedBreed, confidence});

        } else {
            addGalleryItem({src, breed: 'Unknown', confidence: detectionScore});
        }
    } catch (err) {
        console.error('Prediction failed:', err);
        addGalleryItem({src, breed: 'Prediction error', confidence: detectionScore});
    }
}

// start
main();
