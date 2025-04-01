const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const worker = new Worker('worker.js');

// Data buffer for visualization
let positions = [];
const maxPoints = 100; // Limit the number of points to render

// Listen for processed data from the worker
worker.onmessage = (e) => {
    const { x, y, timestamp } = e.data;
    // console.log(`Received aggregated data: x=${x}, y=${y}, timestamp=${timestamp}`);
    positions.push({ x, y, timestamp });
    if (positions.length > maxPoints) positions.shift(); // Keep only the latest 100 points
    render();
};

// Track mouse movement
canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    // Send high-frequency data to worker
    worker.postMessage({ x, y, timestamp: performance.now() });
});

// Rendering function (runs at ~60Hz via requestAnimationFrame)
function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;

    for (let i = 0; i < positions.length; i++) {
        const pos = positions[i];
        if (i === 0) {
            ctx.moveTo(pos.x, pos.y);
        } else {
            ctx.lineTo(pos.x, pos.y);
        }
    }
    ctx.stroke();
}

// Start animation loop
function animate() {
    requestAnimationFrame(animate);
}
animate();