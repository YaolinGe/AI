/**
 * Gibbs Sampling Implementation
 */

(function() {
    // Reference to the utility functions
    const utils = window.utils;
    const distributions = utils.distributions;
    
    // Canvas and animation variables
    let canvas;
    let animationId;
    let samples = [];
    let path = [];
    let totalIterations = 0;
    
    // Sampling parameters
    let correlation = 0.5;
    let stepsPerSample = 1;
    let showPath = true;
    
    // Current state of the Markov chain
    let currentState = { x: 0, y: 0 };
    
    // Sample range
    const xMin = -4;
    const xMax = 4;
    const yMin = -4;
    const yMax = 4;
    
    /**
     * Initialize the Gibbs sampling visualization
     */
    function init() {
        canvas = document.getElementById('visualization-canvas');
        
        // Resize canvas to fit container
        function resizeCanvas() {
            const wrapper = canvas.parentElement;
            canvas.width = wrapper.clientWidth;
            canvas.height = wrapper.clientHeight;
        }
        
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
        
        // Set up controls
        setupControls();
        
        // Draw initial state
        drawInitialState();
        
        // Reset stats
        resetStats();
    }
    
    /**
     * Set up control panel event listeners
     */
    function setupControls() {
        // Correlation control
        const correlationControl = document.getElementById('correlation');
        if (correlationControl) {
            correlationControl.addEventListener('input', function() {
                correlation = parseFloat(this.value);
                document.getElementById('correlation-value').textContent = correlation;
                drawInitialState();
            });
        }
        
        // Steps per sample control
        const stepsControl = document.getElementById('gibbs-steps-per-sample');
        if (stepsControl) {
            stepsControl.addEventListener('input', function() {
                stepsPerSample = parseInt(this.value);
                document.getElementById('gibbs-steps-value').textContent = stepsPerSample;
                drawInitialState();
            });
        }
        
        // Show path checkbox
        const pathControl = document.getElementById('show-path');
        if (pathControl) {
            pathControl.addEventListener('change', function() {
                showPath = this.checked;
                drawInitialState();
            });
        }
        
        // Run button
        const runButton = document.getElementById('run-btn');
        if (runButton) {
            runButton.addEventListener('click', function() {
                // Stop any running animation
                if (animationId) {
                    cancelAnimationFrame(animationId);
                    animationId = null;
                    this.textContent = 'Run Sampling';
                } else {
                    // Start animation
                    startSampling();
                    this.textContent = 'Pause';
                }
            });
        }
        
        // Reset button
        const resetButton = document.getElementById('reset-btn');
        if (resetButton) {
            resetButton.addEventListener('click', function() {
                // Stop animation
                if (animationId) {
                    cancelAnimationFrame(animationId);
                    animationId = null;
                    document.getElementById('run-btn').textContent = 'Run Sampling';
                }
                
                // Reset everything
                samples = [];
                path = [];
                currentState = { x: 0, y: 0 };
                resetStats();
                drawInitialState();
            });
        }
    }
    
    /**
     * Reset statistics counters
     */
    function resetStats() {
        totalIterations = 0;
        
        document.getElementById('acceptance-rate').textContent = '0%';
        document.getElementById('samples-count').textContent = '0';
        document.getElementById('iterations-count').textContent = '0';
        
        // Update legend for efficiency
        document.querySelector('.stats-panel h3').textContent = 'Statistics';
        document.querySelector('#acceptance-rate').parentElement.textContent = 
            'Correlation: ' + document.getElementById('acceptance-rate').outerHTML;
    }
    
    /**
     * Draw the initial state with bivariate distribution and possibly sample path
     */
    function drawInitialState() {
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw background contour plot
        drawBivariateContours();
        
        // Draw sample path
        if (showPath && path.length > 0) {
            drawSamplePath();
        }
        
        // Draw samples
        if (samples.length > 0) {
            drawScatterPlot();
        }
    }
    
    /**
     * Draw the bivariate normal contours
     */
    function drawBivariateContours() {
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Parameters for drawing
        const padding = { top: 20, right: 20, bottom: 30, left: 40 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        
        // Draw axes
        ctx.beginPath();
        ctx.moveTo(padding.left, height - padding.bottom);
        ctx.lineTo(padding.left + chartWidth, height - padding.bottom);
        ctx.strokeStyle = '#666';
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(padding.left, height - padding.bottom);
        ctx.lineTo(padding.left, padding.top);
        ctx.stroke();
        
        // Function to map data coordinates to canvas coordinates
        const mapX = x => padding.left + ((x - xMin) / (xMax - xMin)) * chartWidth;
        const mapY = y => height - padding.bottom - ((y - yMin) / (yMax - yMin)) * chartHeight;
        
        // Draw grid lines
        ctx.strokeStyle = '#e6e6e6';
        
        // Vertical grid lines
        for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x++) {
            if (x === 0) continue; // Skip zero axis
            ctx.beginPath();
            ctx.moveTo(mapX(x), mapY(yMin));
            ctx.lineTo(mapX(x), mapY(yMax));
            ctx.stroke();
        }
        
        // Horizontal grid lines
        for (let y = Math.ceil(yMin); y <= Math.floor(yMax); y++) {
            if (y === 0) continue; // Skip zero axis
            ctx.beginPath();
            ctx.moveTo(mapX(xMin), mapY(y));
            ctx.lineTo(mapX(xMax), mapY(y));
            ctx.stroke();
        }
        
        // Draw contour lines for bivariate normal distribution
        // We'll pre-compute the density on a grid and then use it to draw contours
        const gridSize = 40;
        const xStep = (xMax - xMin) / gridSize;
        const yStep = (yMax - yMin) / gridSize;
        
        const densityGrid = [];
        let maxDensity = 0;
        
        // Compute density values
        for (let i = 0; i <= gridSize; i++) {
            densityGrid[i] = [];
            const x = xMin + i * xStep;
            
            for (let j = 0; j <= gridSize; j++) {
                const y = yMin + j * yStep;
                const density = bivariateNormalPdf(x, y, 0, 0, 1, 1, correlation);
                densityGrid[i][j] = density;
                maxDensity = Math.max(maxDensity, density);
            }
        }
        
        // Draw filled contours
        const contourLevels = 10;
        const colorScale = value => {
            // Rainbow color scale from blue (low) to red (high)
            const hue = (1 - value) * 240; // 240 is blue, 0 is red
            return `hsla(${hue}, 100%, 50%, 0.1)`;
        };
        
        for (let level = contourLevels; level >= 1; level--) {
            const threshold = (level / contourLevels) * maxDensity;
            
            ctx.beginPath();
            
            for (let i = 0; i < gridSize; i++) {
                for (let j = 0; j < gridSize; j++) {
                    const x1 = xMin + i * xStep;
                    const y1 = yMin + j * yStep;
                    const x2 = x1 + xStep;
                    const y2 = y1 + yStep;
                    
                    if (densityGrid[i][j] >= threshold) {
                        ctx.rect(mapX(x1), mapY(y2), mapX(x2) - mapX(x1), mapY(y1) - mapY(y2));
                    }
                }
            }
            
            ctx.fillStyle = colorScale(level / contourLevels);
            ctx.fill();
        }
        
        // Draw contour lines
        const lineContourLevels = 5;
        const lineWidth = 1;
        
        for (let level = 1; level <= lineContourLevels; level++) {
            const threshold = (level / lineContourLevels) * maxDensity * 0.9;
            
            // Simple marching squares algorithm for contours
            for (let i = 0; i < gridSize; i++) {
                for (let j = 0; j < gridSize; j++) {
                    const x1 = xMin + i * xStep;
                    const y1 = yMin + j * yStep;
                    const x2 = x1 + xStep;
                    const y2 = y1 + yStep;
                    
                    const v1 = densityGrid[i][j];
                    const v2 = densityGrid[i+1][j];
                    const v3 = densityGrid[i+1][j+1];
                    const v4 = densityGrid[i][j+1];
                    
                    // Simple cell edge crossing - not a full marching squares implementation
                    if ((v1 < threshold && v2 >= threshold) || (v1 >= threshold && v2 < threshold)) {
                        const t = (threshold - v1) / (v2 - v1);
                        const x = x1 + t * xStep;
                        
                        ctx.beginPath();
                        ctx.moveTo(mapX(x), mapY(y1));
                        ctx.lineTo(mapX(x), mapY(y1) + lineWidth);
                        ctx.strokeStyle = '#666';
                        ctx.stroke();
                    }
                    
                    if ((v1 < threshold && v4 >= threshold) || (v1 >= threshold && v4 < threshold)) {
                        const t = (threshold - v1) / (v4 - v1);
                        const y = y1 + t * yStep;
                        
                        ctx.beginPath();
                        ctx.moveTo(mapX(x1), mapY(y));
                        ctx.lineTo(mapX(x1) + lineWidth, mapY(y));
                        ctx.strokeStyle = '#666';
                        ctx.stroke();
                    }
                }
            }
        }
        
        // Draw axes labels
        ctx.fillStyle = '#333';
        ctx.textAlign = 'center';
        ctx.font = '12px sans-serif';
        
        // X-axis labels
        for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x++) {
            if (x % 2 === 0) { // Show every other label to avoid crowding
                ctx.fillText(x.toString(), mapX(x), height - padding.bottom + 15);
            }
        }
        
        // Y-axis labels
        ctx.textAlign = 'right';
        for (let y = Math.ceil(yMin); y <= Math.floor(yMax); y++) {
            if (y % 2 === 0) { // Show every other label to avoid crowding
                ctx.fillText(y.toString(), padding.left - 5, mapY(y) + 4);
            }
        }
        
        // Title
        ctx.textAlign = 'center';
        ctx.font = '14px sans-serif';
        ctx.fillText(`Bivariate Normal Distribution (ρ = ${correlation})`, width / 2, padding.top - 5);
    }
    
    /**
     * Draw the Gibbs sampling path
     */
    function drawSamplePath() {
        if (!canvas || path.length === 0) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Parameters for drawing
        const padding = { top: 20, right: 20, bottom: 30, left: 40 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        
        // Function to map data coordinates to canvas coordinates
        const mapX = x => padding.left + ((x - xMin) / (xMax - xMin)) * chartWidth;
        const mapY = y => height - padding.bottom - ((y - yMin) / (yMax - yMin)) * chartHeight;
        
        // Draw path lines
        ctx.beginPath();
        ctx.moveTo(mapX(path[0].x), mapY(path[0].y));
        
        for (let i = 1; i < path.length; i++) {
            // Draw lines in a way to highlight the Gibbs pattern (alternating x and y updates)
            if (i % 2 === 1) {
                // First update x, keeping y constant
                ctx.lineTo(mapX(path[i].x), mapY(path[i-1].y));
                ctx.lineTo(mapX(path[i].x), mapY(path[i].y));
            } else {
                // Then update y, keeping x constant
                ctx.lineTo(mapX(path[i-1].x), mapY(path[i].y));
                ctx.lineTo(mapX(path[i].x), mapY(path[i].y));
            }
        }
        
        ctx.strokeStyle = 'rgba(52, 152, 219, 0.7)'; // Semi-transparent blue
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        // Draw current position
        const currentX = mapX(path[path.length - 1].x);
        const currentY = mapY(path[path.length - 1].y);
        
        ctx.beginPath();
        ctx.arc(currentX, currentY, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#e74c3c'; // Red
        ctx.fill();
        ctx.strokeStyle = '#c0392b';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    
    /**
     * Draw scatter plot of accepted samples
     */
    function drawScatterPlot() {
        if (!canvas || samples.length === 0) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Parameters for drawing
        const padding = { top: 20, right: 20, bottom: 30, left: 40 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        
        // Function to map data coordinates to canvas coordinates
        const mapX = x => padding.left + ((x - xMin) / (xMax - xMin)) * chartWidth;
        const mapY = y => height - padding.bottom - ((y - yMin) / (yMax - yMin)) * chartHeight;
        
        // Draw samples as points
        for (const sample of samples) {
            ctx.beginPath();
            ctx.arc(mapX(sample.x), mapY(sample.y), 2, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(46, 204, 113, 0.7)'; // Semi-transparent green
            ctx.fill();
        }
    }
    
    /**
     * Bivariate normal distribution PDF
     */
    function bivariateNormalPdf(x, y, meanX = 0, meanY = 0, sigmaX = 1, sigmaY = 1, rho = 0) {
        const z = Math.pow((x - meanX) / sigmaX, 2) - 
                 2 * rho * ((x - meanX) / sigmaX) * ((y - meanY) / sigmaY) + 
                 Math.pow((y - meanY) / sigmaY, 2);
        
        const normalizer = 2 * Math.PI * sigmaX * sigmaY * Math.sqrt(1 - rho * rho);
        
        return Math.exp(-z / (2 * (1 - rho * rho))) / normalizer;
    }
    
    /**
     * Conditional distribution for X given Y (using properties of bivariate normal)
     */
    function conditionalX(y, meanX = 0, meanY = 0, sigmaX = 1, sigmaY = 1, rho = 0) {
        const conditionalMean = meanX + rho * (sigmaX / sigmaY) * (y - meanY);
        const conditionalVariance = sigmaX * sigmaX * (1 - rho * rho);
        
        return {
            mean: conditionalMean,
            stdDev: Math.sqrt(conditionalVariance),
            sample: () => distributions.normal.sample(conditionalMean, Math.sqrt(conditionalVariance))
        };
    }
    
    /**
     * Conditional distribution for Y given X (using properties of bivariate normal)
     */
    function conditionalY(x, meanX = 0, meanY = 0, sigmaX = 1, sigmaY = 1, rho = 0) {
        const conditionalMean = meanY + rho * (sigmaY / sigmaX) * (x - meanX);
        const conditionalVariance = sigmaY * sigmaY * (1 - rho * rho);
        
        return {
            mean: conditionalMean,
            stdDev: Math.sqrt(conditionalVariance),
            sample: () => distributions.normal.sample(conditionalMean, Math.sqrt(conditionalVariance))
        };
    }
    
    /**
     * Start the Gibbs sampling process
     */
    function startSampling() {
        if (!canvas) return;
        
        // Animation function
        function animate() {
            // Perform a batch of sampling steps
            const batchSize = 5;
            for (let i = 0; i < batchSize; i++) {
                performGibbsStep();
            }
            
            // Update visualization
            drawInitialState();
            
            // Continue animation
            animationId = requestAnimationFrame(animate);
        }
        
        animate();
    }
    
    /**
     * Perform a single Gibbs sampling step
     */
    function performGibbsStep() {
        // X and Y means (both 0 for standard bivariate normal)
        const meanX = 0;
        const meanY = 0;
        
        // Standard deviations (both 1 for standard bivariate normal)
        const sigmaX = 1;
        const sigmaY = 1;
        
        // Track path
        path.push({...currentState});
        
        // Perform Gibbs step: sample x conditioned on y, then y conditioned on x
        // First update x
        const condX = conditionalX(currentState.y, meanX, meanY, sigmaX, sigmaY, correlation);
        currentState.x = condX.sample();
        
        // Add intermediate state to path to show Gibbs pattern
        path.push({...currentState});
        
        // Then update y
        const condY = conditionalY(currentState.x, meanX, meanY, sigmaX, sigmaY, correlation);
        currentState.y = condY.sample();
        
        // Update iteration count
        totalIterations++;
        
        // Add sample after specified number of steps
        if (totalIterations % stepsPerSample === 0) {
            samples.push({...currentState});
        }
        
        // Update statistics display
        document.getElementById('acceptance-rate').textContent = correlation.toFixed(1);
        document.getElementById('samples-count').textContent = samples.length;
        document.getElementById('iterations-count').textContent = totalIterations;
    }
    
    // Export the module
    window.gibbsSampling = {
        init: init
    };
})(); 