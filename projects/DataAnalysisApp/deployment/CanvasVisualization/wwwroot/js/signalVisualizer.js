let canvas;
let ctx;
let animationFrameId;
let dataBuffer = [];
const maxBufferSize = 1000;

export function initializeCanvas(canvasElement) {
    canvas = canvasElement;
    ctx = canvas.getContext('2d');
    
    // Set canvas size to match display size
    const resizeCanvas = () => {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
    };
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
}

export function updateSignal(data, windowSize) {
    // Add new data to buffer
    dataBuffer.push(...data);
    
    // Keep buffer size manageable
    if (dataBuffer.length > maxBufferSize) {
        dataBuffer = dataBuffer.slice(-maxBufferSize);
    }
    
    // Cancel any pending animation frame
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    
    // Schedule new render
    animationFrameId = requestAnimationFrame(() => renderSignal(windowSize));
}

function renderSignal(windowSize) {
    if (!ctx || !canvas) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate visible data points
    const visibleData = dataBuffer.slice(-windowSize);
    if (visibleData.length === 0) return;
    
    // Calculate scaling factors
    const xScale = canvas.width / (windowSize - 1);
    const yScale = canvas.height / 2;
    const yOffset = canvas.height / 2;
    
    // Draw grid
    drawGrid();
    
    // Draw signal
    ctx.beginPath();
    ctx.strokeStyle = '#2196F3';
    ctx.lineWidth = 2;
    
    visibleData.forEach((value, index) => {
        const x = index * xScale;
        const y = yOffset - (value * yScale);
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    
    ctx.stroke();
}

function drawGrid() {
    const gridSize = 50;
    ctx.strokeStyle = '#E0E0E0';
    ctx.lineWidth = 0.5;
    
    // Vertical lines
    for (let x = 0; x <= canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    
    // Horizontal lines
    for (let y = 0; y <= canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
    
    // Draw axes
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, canvas.height);
    ctx.stroke();
} 