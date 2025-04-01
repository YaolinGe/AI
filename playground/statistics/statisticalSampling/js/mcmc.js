/**
 * Markov Chain Monte Carlo (MCMC) Implementation
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
    let proposalWidth = 1.0;
    let burnInPeriod = 100;
    
    // Current state of the Markov chain
    let currentState = 0;
    
    // Sample range
    const xMin = -10;
    const xMax = 10;
    
    /**
     * Initialize the MCMC visualization
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
        const targetSelect = document.getElementById('target-dist-mcmc');
        if (targetSelect) {
            targetSelect.addEventListener('change', function() {
                targetDist = this.value;
                drawInitialState();
            });
        }
        
        // Proposal width control
        const widthControl = document.getElementById('proposal-width');
        if (widthControl) {
            widthControl.addEventListener('input', function() {
                proposalWidth = parseFloat(this.value);
                document.getElementById('proposal-width-value').textContent = proposalWidth.toFixed(1);
                drawInitialState();
            });
        }
        
        // Burn-in period control
        const burnInControl = document.getElementById('burn-in');
        if (burnInControl) {
            burnInControl.addEventListener('input', function() {
                burnInPeriod = parseInt(this.value);
                document.getElementById('burn-in-value').textContent = burnInPeriod;
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
     * Draw the initial state with distributions
     */
    function drawInitialState() {
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw target distribution
        drawDistribution();
        
        // Draw Markov chain trajectory and samples
        if (trajectory.length > 0) {
            drawTrajectory();
        }
        
        // Draw samples histogram if any
        if (samples.length > 0) {
            drawHistogram();
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
        const chartHeight = height - padding.top - padding.bottom - 100; // Leave space for histogram
        
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
        
        // Add legend
        ctx.font = '12px sans-serif';
        
        // Target distribution
        ctx.fillStyle = '#e74c3c';
        ctx.fillRect(padding.left, padding.top, 15, 10);
        ctx.fillStyle = '#333';
        ctx.fillText('Target Distribution', padding.left + 20, padding.top + 10);
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
        const trajectoryY = height - padding.bottom - 100 - distributionHeight * 0.05;
        
        // Draw trajectory line
        ctx.beginPath();
        ctx.moveTo(mapX(trajectory[0]), trajectoryY);
        
        for (let i = 1; i < trajectory.length; i++) {
            ctx.lineTo(mapX(trajectory[i]), trajectoryY);
        }
        
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw burn-in period marker if applicable
        if (burnInPeriod > 0 && trajectory.length > burnInPeriod) {
            const burnInX = mapX(trajectory[burnInPeriod]);
            
            ctx.beginPath();
            ctx.moveTo(burnInX, trajectoryY - 5);
            ctx.lineTo(burnInX, trajectoryY + 5);
            ctx.strokeStyle = '#e67e22';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Add burn-in label
            ctx.fillStyle = '#e67e22';
            ctx.textAlign = 'center';
            ctx.fillText('Burn-in End', burnInX, trajectoryY - 10);
        }
        
        // Draw current position
        if (trajectory.length > 0) {
            const currentX = mapX(trajectory[trajectory.length - 1]);
            
            ctx.beginPath();
            ctx.arc(currentX, trajectoryY, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#2ecc71';
            ctx.fill();
            ctx.strokeStyle = '#27ae60';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }
    
    /**
     * Draw histogram of accepted samples (after burn-in)
     */
    function drawHistogram() {
        if (!canvas || samples.length === 0) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Parameters for drawing
        const padding = { top: 20, right: 20, bottom: 30, left: 40 };
        const histogramTop = height - padding.bottom - 90;
        const histogramHeight = 80;
        const chartWidth = width - padding.left - padding.right;
        
        // Number of bins for the histogram
        const bins = 30;
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
        
        // Find maximum bin count for scaling
        const maxCount = Math.max(...binCounts);
        
        // Draw histogram bars
        ctx.fillStyle = '#2ecc71'; // Green for histogram
        
        for (let i = 0; i < bins; i++) {
            const barHeight = (binCounts[i] / maxCount) * histogramHeight;
            const barX = padding.left + (i * chartWidth / bins);
            const barY = histogramTop - barHeight;
            
            ctx.fillRect(
                barX,
                barY,
                chartWidth / bins - 1,
                barHeight
            );
        }
        
        // Draw x-axis for histogram
        ctx.beginPath();
        ctx.moveTo(padding.left, histogramTop);
        ctx.lineTo(padding.left + chartWidth, histogramTop);
        ctx.strokeStyle = '#666';
        ctx.stroke();
        
        // Labels for histogram
        ctx.fillStyle = '#333';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Histogram of Accepted Samples (After Burn-in)', width / 2, histogramTop + 20);
    }
    
    /**
     * Start the MCMC sampling process
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
     * Perform a single MCMC sampling step
     */
    function performSamplingStep() {
        // Get the target distribution function
        const targetFunction = distributions[targetDist].pdf;
        
        // Propose a new state using a symmetric proposal (normal distribution centered at current state)
        const proposedState = currentState + distributions.normal.sample(0, proposalWidth);
        
        // Evaluate target density at current and proposed states
        const currentDensity = targetFunction(currentState);
        const proposedDensity = targetFunction(proposedState);
        
        // Calculate acceptance probability
        const acceptanceProb = Math.min(1, proposedDensity / currentDensity);
        
        // Generate uniform random number for acceptance test
        const u = Math.random();
        
        // Record the current state in the trajectory
        trajectory.push(currentState);
        
        // Accept or reject the proposed state
        if (u <= acceptanceProb) {
            currentState = proposedState;
            acceptanceCount++;
            
            // If we're past the burn-in period, add to samples
            if (trajectory.length > burnInPeriod) {
                samples.push(currentState);
            }
        }
        
        // Update iteration count
        totalIterations++;
        
        // Update statistics display
        const acceptanceRate = (acceptanceCount / totalIterations * 100).toFixed(1);
        document.getElementById('acceptance-rate').textContent = acceptanceRate + '%';
        document.getElementById('samples-count').textContent = samples.length;
        document.getElementById('iterations-count').textContent = totalIterations;
    }
    
    // Export the module
    window.mcmc = {
        init: init
    };
})(); 