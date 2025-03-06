let canvas, ctx;
let running = false;
let signalType = 'sine';
let windowSize = 10;
let speed = 1000;
let signalData = [];
let previousTime = Date.now();

function initializeCanvas(canvasId) {
    canvas = document.getElementById(canvasId);
    ctx = canvas.getContext('2d');
}

function startSignalStreaming(type, avgWindowSize, rate) {
    signalType = type || 'sine';
    windowSize = avgWindowSize || 10;
    speed = rate || 1000;
    signalData = [];
    running = true;
    previousTime = Date.now();
    streamSignal();
}

function stopSignalStreaming() {
    running = false;
}

function streamSignal() {
    if (!running) return;

    let currentTime = Date.now();
    let timeDiff = currentTime - previousTime;

    if (timeDiff >= (1000 / speed)) {
        // Generate the signal data based on the selected type
        let newSignalValue = generateSignal(currentTime);
        signalData.push(newSignalValue);

        // Apply moving average
        if (signalData.length > windowSize) {
            signalData.shift(); // Remove the oldest value
        }

        // Draw the signal on canvas
        drawSignal();

        previousTime = currentTime;
    }

    requestAnimationFrame(streamSignal);
}

function generateSignal(time) {
    switch (signalType) {
        case 'sine':
            return Math.sin(time / 1000 * 2 * Math.PI);  // Sine wave signal
        case 'cosine':
            return Math.cos(time / 1000 * 2 * Math.PI); // Cosine wave signal
        case 'gaussian':
            return Math.random() * 2 - 1;  // Gaussian noise (rough approximation)
        case 'random':
            return Math.random() * 2 - 1;  // Random signal
        default:
            return 0;
    }
}

function drawSignal() {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the signal on the canvas
    ctx.beginPath();
    let x = 0;
    signalData.forEach(value => {
        let y = (value + 1) * (canvas.height / 2);  // Normalize value to canvas height
        ctx.lineTo(x, y);
        x += canvas.width / windowSize;
    });
    ctx.stroke();
}
