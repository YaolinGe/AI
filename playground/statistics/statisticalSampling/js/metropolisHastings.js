/**
 * Metropolis-Hastings Algorithm Implementation
 */

(function() {
    // Reference to the utility functions
    const utils = window.utils;
    const distributions = utils.distributions;
    
    // Canvas and animation variables
    let canvas;
    let animationId;
    let samples = [];
    let trajectory = [];
    let acceptanceCount = 0;
    let totalIterations = 0;
    
    // Sampling parameters
    let targetDist = 'bimodal';
    let proposalStdDev = 1.0;
    let showTrajectory = true;
    
    // Current state of the Markov chain
    let currentState = 0;
    
    // Sample range
    const xMin = -10;
    const xMax = 10;
    
    /**
     * Initialize the Metropolis-Hastings visualization
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
        // Target distribution selection
        const targetSelect = document.getElementById('target-dist-mh');
        if (targetSelect) {
            targetSelect.addEventListener('change', function() {
                targetDist = this.value;
                drawInitialState();
            });
        }
        
        // Proposal std dev control
        const stdDevControl = document.getElementById('proposal-std-mh');
        if (stdDevControl) {
            stdDevControl.addEventListener('input', function() {
                proposalStdDev = parseFloat(this.value);
                document.getElementById('proposal-std-mh-value').textContent = proposalStdDev.toFixed(1);
                drawInitialState();
            });
        }
        
        // Show trajectory checkbox
        const trajectoryControl = document.getElementById('show-trajectory');
        if (trajectoryControl) {
            trajectoryControl.addEventListener('change', function() {
                showTrajectory = this.checked;
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
                trajectory = [];
                currentState = 0;
                resetStats();
                drawInitialState();
            });
        }
    }
    
    /**
     * Reset statistics counters
     */
    function resetStats() {
        acceptanceCount = 0;
        totalIterations = 0;
        
        document.getElementById('acceptance-rate').textContent = '0%';
        document.getElementById('samples-count').textContent = '0';
        document.getElementById('iterations-count').textContent = '0';
    }
    
    /**
     * Draw the initial state with distribution and possibly density plot
     */
    function drawInitialState() {
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw target distribution
        drawDistribution();
        
        // Draw Markov chain trajectory
        if (showTrajectory && trajectory.length > 0) {
            drawTrajectory();
        }
        
        // Draw density plot if we have enough samples
        if (samples.length > 0) {
            drawDensityPlot();
        }
    }
    
    /**
     * Draw the target distribution
     */
    function drawDistribution() {
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Parameters for drawing
        const padding = { top: 20, right: 20, bottom: 30, left: 40 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom - 100; // Leave space for density plot
        
        // Draw x-axis
        ctx.beginPath();
        ctx.moveTo(padding.left, height - padding.bottom - 100);
        ctx.lineTo(padding.left + chartWidth, height - padding.bottom - 100);
        ctx.strokeStyle = '#666';
        ctx.stroke();
        
        // Sample points for drawing distribution
        const points = 200;
        const dx = (xMax - xMin) / points;
        
        // Get target distribution function
        const targetFunction = distributions[targetDist].pdf;
        
        // Calculate maximum value to scale
        let maxTargetVal = 0;
        
        for (let i = 0; i <= points; i++) {
            const x = xMin + i * dx;
            const targetVal = targetFunction(x);
            maxTargetVal = Math.max(maxTargetVal, targetVal);
        }
        
        // Scale factor for visualization
        const scaleFactor = maxTargetVal > 0 ? 1 / maxTargetVal : 1;
        
        // Draw target distribution
        ctx.beginPath();
        
        for (let i = 0; i <= points; i++) {
            const x = xMin + i * dx;
            const xPixel = padding.left + ((x - xMin) / (xMax - xMin)) * chartWidth;
            const y = targetFunction(x) * scaleFactor * chartHeight;
            const yPixel = height - padding.bottom - 100 - y;
            
            if (i === 0) {
                ctx.moveTo(xPixel, yPixel);
            } else {
                ctx.lineTo(xPixel, yPixel);
            }
        }
        
        ctx.strokeStyle = '#e74c3c'; // Target distribution color
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw the current proposal distribution (normal centered at current state)
        if (trajectory.length > 0) {
            const currentX = trajectory[trajectory.length - 1];
            
            ctx.beginPath();
            
            for (let i = 0; i <= points; i++) {
                const x = xMin + i * dx;
                const proposalDensity = distributions.normal.pdf(x, currentX, proposalStdDev);
                // Scale the proposal to be visible
                const maxScale = maxTargetVal / distributions.normal.pdf(currentX, currentX, proposalStdDev);
                const y = proposalDensity * scaleFactor * chartHeight * maxScale * 0.5;
                const xPixel = padding.left + ((x - xMin) / (xMax - xMin)) * chartWidth;
                const yPixel = height - padding.bottom - 100 - y;
                
                if (i === 0) {
                    ctx.moveTo(xPixel, yPixel);
                } else {
                    ctx.lineTo(xPixel, yPixel);
                }
            }
            
            ctx.strokeStyle = 'rgba(52, 152, 219, 0.5)'; // Proposal distribution color (semi-transparent blue)
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
        
        // Add legend
        ctx.font = '12px sans-serif';
        
        // Target distribution
        ctx.fillStyle = '#e74c3c';
        ctx.fillRect(padding.left, padding.top, 15, 10);
        ctx.fillStyle = '#333';
        ctx.fillText('Target Distribution', padding.left + 20, padding.top + 10);
        
        // Proposal distribution (if shown)
        if (trajectory.length > 0) {
            ctx.fillStyle = 'rgba(52, 152, 219, 0.5)';
            ctx.fillRect(padding.left + 150, padding.top, 15, 10);
            ctx.fillStyle = '#333';
            ctx.fillText('Current Proposal Distribution', padding.left + 170, padding.top + 10);
        }
    }
    
    /**
     * Draw the Markov chain trajectory
     */
    function drawTrajectory() {
        if (!canvas || trajectory.length === 0) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Parameters for drawing
        const padding = { top: 20, right: 20, bottom: 30, left: 40 };
        const chartWidth = width - padding.left - padding.right;
        const distributionHeight = height - padding.top - padding.bottom - 100;
        
        // Function to map x coordinate to pixel position
        const mapX = x => padding.left + ((x - xMin) / (xMax - xMin)) * chartWidth;
        
        // Calculate y position for trajectory
        const trajectoryY = height - padding.bottom - 100 - distributionHeight * 0.1;
        
        // Draw trajectory line
        ctx.beginPath();
        ctx.moveTo(mapX(trajectory[0]), trajectoryY);
        
        for (let i = 1; i < trajectory.length; i++) {
            // If there was an acceptance (state changed), draw a vertical line
            if (trajectory[i] !== trajectory[i-1]) {
                ctx.lineTo(mapX(trajectory[i]), trajectoryY);
                ctx.strokeStyle = '#27ae60'; // Green for accepted moves
            } else {
                ctx.lineTo(mapX(trajectory[i]), trajectoryY);
                ctx.strokeStyle = '#c0392b'; // Red for rejected moves
            }
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(mapX(trajectory[i]), trajectoryY);
        }
        
        // Draw current position
        const currentX = mapX(trajectory[trajectory.length - 1]);
        
        ctx.beginPath();
        ctx.arc(currentX, trajectoryY, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#3498db';
        ctx.fill();
        ctx.strokeStyle = '#2980b9';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    
    /**
     * Draw density plot from samples
     */
    function drawDensityPlot() {
        if (!canvas || samples.length === 0) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Parameters for drawing
        const padding = { top: 20, right: 20, bottom: 30, left: 40 };
        const densityTop = height - padding.bottom - 90;
        const densityHeight = 80;
        const chartWidth = width - padding.left - padding.right;
        
        // Number of bins for the density plot
        const bins = 50;
        const binWidth = (xMax - xMin) / bins;
        
        // Initialize bins counts
        const binCounts = new Array(bins).fill(0);
        
        // Count samples in each bin
        samples.forEach(sample => {
            const binIndex = Math.min(Math.floor((sample - xMin) / binWidth), bins - 1);
            if (binIndex >= 0 && binIndex < bins) {
                binCounts[binIndex]++;
            }
        });
        
        // Apply kernel smoothing (simple moving average)
        const smoothedCounts = [];
        const kernelWidth = 3;
        
        for (let i = 0; i < bins; i++) {
            let sum = 0;
            let count = 0;
            
            for (let j = Math.max(0, i - kernelWidth); j <= Math.min(bins - 1, i + kernelWidth); j++) {
                sum += binCounts[j];
                count++;
            }
            
            smoothedCounts.push(sum / count);
        }
        
        // Find maximum bin count for scaling
        const maxCount = Math.max(...smoothedCounts);
        
        // Draw density curve
        ctx.beginPath();
        
        for (let i = 0; i < bins; i++) {
            const x = padding.left + (i * chartWidth / bins);
            const y = densityTop - (smoothedCounts[i] / maxCount) * densityHeight;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.strokeStyle = '#2ecc71';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Fill area under the curve
        ctx.lineTo(padding.left + chartWidth, densityTop);
        ctx.lineTo(padding.left, densityTop);
        ctx.closePath();
        ctx.fillStyle = 'rgba(46, 204, 113, 0.2)';
        ctx.fill();
        
        // Draw x-axis for density plot
        ctx.beginPath();
        ctx.moveTo(padding.left, densityTop);
        ctx.lineTo(padding.left + chartWidth, densityTop);
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Labels for density plot
        ctx.fillStyle = '#333';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Kernel Density Estimate of Samples', width / 2, densityTop + 20);
    }
    
    /**
     * Start the Metropolis-Hastings sampling process
     */
    function startSampling() {
        if (!canvas) return;
        
        // Animation function
        function animate() {
            // Perform a batch of sampling steps
            const batchSize = 5;
            for (let i = 0; i < batchSize; i++) {
                performSamplingStep();
            }
            
            // Update visualization
            drawInitialState();
            
            // Continue animation
            animationId = requestAnimationFrame(animate);
        }
        
        animate();
    }
    
    /**
     * Perform a single Metropolis-Hastings sampling step
     */
    function performSamplingStep() {
        // Get the target distribution function
        const targetFunction = distributions[targetDist].pdf;
        
        // Propose a new state using a symmetric proposal (normal distribution centered at current state)
        const proposedState = currentState + distributions.normal.sample(0, proposalStdDev);
        
        // Evaluate target density at current and proposed states
        const currentDensity = targetFunction(currentState);
        const proposedDensity = targetFunction(proposedState);
        
        // Calculate acceptance probability
        // For symmetric proposal distributions (like normal centered at current state), 
        // this simplifies to the ratio of target densities
        const acceptanceProb = Math.min(1, proposedDensity / currentDensity);
        
        // Generate uniform random number for acceptance test
        const u = Math.random();
        
        // Record the current state in the trajectory
        trajectory.push(currentState);
        
        // Accept or reject the proposed state
        if (u <= acceptanceProb) {
            currentState = proposedState;
            acceptanceCount++;
        }
        
        // Always add the current state to samples (after thinning and burn-in if desired)
        samples.push(currentState);
        
        // Update iteration count
        totalIterations++;
        
        // Update statistics display
        const acceptanceRate = (acceptanceCount / totalIterations * 100).toFixed(1);
        document.getElementById('acceptance-rate').textContent = acceptanceRate + '%';
        document.getElementById('samples-count').textContent = samples.length;
        document.getElementById('iterations-count').textContent = totalIterations;
    }
    
    // Export the module
    window.metropolisHastings = {
        init: init
    };
})(); 