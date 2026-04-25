(() => {
    let pointMode = null;
    let currentFeed = 'static';
    let currentFrameNum = 1;

    const feed = document.getElementById('dashboard-feed');
    const summary = document.getElementById('summary');
    const mode = document.getElementById('mode');
    const paths = document.getElementById('paths');
    const frameIndicator = document.getElementById('frame-indicator');

    function changeImage(imageName) {
        feed.src = `/process_image?img=${imageName}&t=${Date.now()}`;
        currentFeed = 'static';
    }

    function useLiveFeed() {
        feed.src = '/video_feed';
        currentFeed = 'live';
    }

    function setPointMode(nextMode) {
        pointMode = nextMode;
        mode.innerText = `Point mode: ${nextMode}`;
    }

    async function saveManualPoint(kind, x, y) {
        await fetch('/api/manual_points', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ [`${kind}_point`]: [Math.round(x), Math.round(y)] }),
        });
    }

    async function refreshSummary() {
        try {
            const res = await fetch('/api/slot_summary');
            const data = await res.json();
            summary.innerText = `Status: ${data.status} | Free: ${data.free_slots} | Occupied: ${data.occupied_slots} | Unknown: ${data.unknown_slots} | FPS: ${data.fps}`;
        } catch (_e) {
            summary.innerText = 'Status API unavailable';
        }
    }

    async function refreshPaths() {
        try {
            const res = await fetch('/api/path_details');
            const data = await res.json();
            if (!data.paths || data.paths.length === 0) {
                paths.innerText = 'No active routed vehicles.';
                return;
            }
            const lines = data.paths.map((p) => {
                const dims = p.car_dimensions || {};
                const steps = (p.instructions || []).join(' -> ');
                return `Track ${p.track_id} | Target: ${p.target} | Car(px): L=${dims.length_px || 0}, W=${dims.width_px || 0} | Route: ${steps}`;
            });
            paths.innerText = lines.join('\n');
        } catch (_e) {
            paths.innerText = 'Path details API unavailable';
        }
    }

    // --- FRAME NAVIGATION LOGIC ---
    function loadCurrentFrame() {
        changeImage(`frame${currentFrameNum}.png`);
        frameIndicator.innerText = `Frame ${currentFrameNum}`;
    }

    document.getElementById('btn-prev').addEventListener('click', () => {
        if (currentFrameNum > 1) {
            currentFrameNum--;
            loadCurrentFrame();
        }
    });

    document.getElementById('btn-next').addEventListener('click', () => {
        currentFrameNum++;
        loadCurrentFrame();
    });

    // --- CORE EVENT LISTENERS ---
    document.getElementById('btn-baseline').addEventListener('click', () => {
        changeImage('baseline.png');
        frameIndicator.innerText = "Baseline";
    });
    document.getElementById('btn-live').addEventListener('click', () => {
        useLiveFeed();
        frameIndicator.innerText = "LIVE";
    });
    document.getElementById('btn-set-entry').addEventListener('click', () => setPointMode('entry'));
    document.getElementById('btn-set-exit').addEventListener('click', () => setPointMode('exit'));

    // --- CLICK TO SET POINTS ---
    feed.addEventListener('click', async (ev) => {
        if (!pointMode) return;
        const rect = feed.getBoundingClientRect();
        const scaleX = feed.naturalWidth / rect.width;
        const scaleY = feed.naturalHeight / rect.height;
        const x = (ev.clientX - rect.left) * scaleX;
        const y = (ev.clientY - rect.top) * scaleY;
        
        await saveManualPoint(pointMode, x, y);
        mode.innerText = `${pointMode} point saved at (${Math.round(x)}, ${Math.round(y)})`;
        pointMode = null;
        
        if (currentFeed === 'live') {
            useLiveFeed();
        } else {
            const base = feed.src.includes('&t=') ? feed.src.split('&t=')[0] : feed.src;
            feed.src = `${base}&t=${Date.now()}`;
        }
    });

    // --- INITIALIZATION ---
    refreshSummary();
    refreshPaths();
    setInterval(refreshSummary, 2000);
    setInterval(refreshPaths, 2000);
})();