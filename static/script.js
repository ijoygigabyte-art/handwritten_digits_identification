// ── Canvas Setup ────────────────────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Drawing state
let isDrawing = false;
let lastX = 0, lastY = 0;

// Scale mouse coords to canvas internal resolution
function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}

// Draw a smooth line segment
function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const { x, y } = getPos(e);
    ctx.lineWidth = 14;      // Thinner stroke → matches MNIST digit thickness when scaled to 28x28
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    [lastX, lastY] = [x, y];
}

// Event listeners for mouse
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const { x, y } = getPos(e);
    [lastX, lastY] = [x, y];
});
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseleave', () => isDrawing = false);

// Event listeners for touch (mobile)
canvas.addEventListener('touchstart', (e) => {
    isDrawing = true;
    const { x, y } = getPos(e);
    [lastX, lastY] = [x, y];
});
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', () => isDrawing = false);


// ── Clear Button ─────────────────────────────────────────────────────────────
document.getElementById('clearBtn').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('digitDisplay').textContent = '?';
    document.getElementById('confidenceDisplay').textContent = '';
    document.getElementById('barChart').innerHTML = '';
});


// ── Predict Button ─────────────────────────────────────────────────────────── 
document.getElementById('predictBtn').addEventListener('click', async () => {
    const btn = document.getElementById('predictBtn');
    btn.textContent = 'Thinking…';
    btn.disabled = true;

    // Export the canvas as a base64 PNG and send to the backend
    const imageData = canvas.toDataURL('image/png');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) throw new Error('Prediction failed');
        const result = await response.json();
        displayResult(result);
    } catch (err) {
        document.getElementById('digitDisplay').textContent = '!';
        document.getElementById('confidenceDisplay').textContent = 'Error — is the server running?';
    } finally {
        btn.textContent = 'Predict →';
        btn.disabled = false;
    }
});


// ── Display Result ────────────────────────────────────────────────────────────
function displayResult({ digit, confidence, probabilities }) {
    document.getElementById('digitDisplay').textContent = digit;
    document.getElementById('confidenceDisplay').textContent = `${confidence}% confident`;

    const chart = document.getElementById('barChart');
    chart.innerHTML = '';

    probabilities.forEach((pct, i) => {
        const isTop = i === digit;
        const row = document.createElement('div');
        row.className = 'bar-row';
        row.innerHTML = `
            <span class="bar-label">${i}</span>
            <div class="bar-track">
                <div class="bar-fill ${isTop ? 'top' : ''}" style="width: ${pct}%"></div>
            </div>
            <span class="bar-pct">${pct}%</span>
        `;
        chart.appendChild(row);
    });
}
