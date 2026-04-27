(() => {
    let pointMode = null;
    let currentFeed = 'static';
    let currentFrameNum = 1;
    let pollInterval = null; // NEW: Tracks our auto-refresh timer

    const feed = document.getElementById('dashboard-feed');
    const summary = document.getElementById('summary');
    const mode = document.getElementById('mode');
    const paths = document.getElementById('paths');
    const frameIndicator = document.getElementById('frame-indicator');

    // NEW: Stop auto-refreshing when we switch away from Live Polling
    function stopPolling() {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    }

    // NEW: Start auto-refreshing the latest.jpg image every 2 seconds
    function startPolling() {
        currentFeed = 'poll';
        stopPolling(); 
        pollInterval = setInterval(() => {
            feed.src = `/process_image?img=latest.jpg&t=${Date.now()}`;
        }, 2000);
    }

    function changeImage(imageName) {
        stopPolling(); // Stop polling if we go back to static frames
        feed.src = `/process_image?img=${imageName}&t=${Date.now()}`;
        currentFeed = 'static';
    }

    function useLiveFeed() {
        stopPolling(); // Stop polling if we switch to raw RTSP feed
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

    document.getElementById('btn-baseline').addEventListener('click', () => {
        changeImage('baseline.png');
        frameIndicator.innerText = "Baseline";
    });
    
    // UPDATED: This button now triggers the auto-refresh loop
    document.getElementById('btn-live-poll').addEventListener('click', () => {
        startPolling(); 
        frameIndicator.innerText = "Live Polling";
        // Force an immediate load before the first interval triggers
        feed.src = `/process_image?img=latest.jpg&t=${Date.now()}`; 
    });

    document.getElementById('btn-live').addEventListener('click', () => {
        useLiveFeed();
        frameIndicator.innerText = "LIVE";
    });

    document.getElementById('btn-set-entry').addEventListener('click', () => setPointMode('entry'));
    document.getElementById('btn-set-exit').addEventListener('click', () => setPointMode('exit'));

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
        } else if (currentFeed === 'poll') {
            feed.src = `/process_image?img=latest.jpg&t=${Date.now()}`;
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

    // AUTO-START LOGIC
    if (IS_LIVE_MODE) {
        // If the server is in Live Mode, auto-click the polling button
        document.getElementById('btn-live-poll').click();
    } else {
        // If the server is in Static Mode, load Frame 1
        loadCurrentFrame();
    }
})();