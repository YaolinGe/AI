// Buffer to store high-frequency data
let dataBuffer = [];
const aggregationInterval = 1; // Approx 60Hz (1000ms / 60 ≈ 16ms)

// Process incoming mouse data
self.onmessage = (e) => {
    const { x, y, timestamp } = e.data;
    dataBuffer.push({ x, y, timestamp });
};

// Aggregate and downsample data every 16ms
setInterval(() => {
    if (dataBuffer.length > 0) {
        // Simple aggregation: take the average position
        const avgX = dataBuffer.reduce((sum, d) => sum + d.x, 0) / dataBuffer.length;
        const avgY = dataBuffer.reduce((sum, d) => sum + d.y, 0) / dataBuffer.length;
        const latestTimestamp = dataBuffer[dataBuffer.length - 1].timestamp;

        // Send aggregated data to main thread
        self.postMessage({ x: avgX, y: avgY, timestamp: latestTimestamp });
        // console.log(`Aggregated data: x=${avgX}, y=${avgY}, timestamp=${latestTimestamp}`);

        // Clear buffer
        dataBuffer = [];
    }
}, aggregationInterval);