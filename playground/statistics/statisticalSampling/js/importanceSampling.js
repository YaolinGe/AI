/**
 * Importance Sampling Implementation
 */

(function() {
    // Reference to the utility functions
    const utils = window.utils;
    const distributions = utils.distributions;
    
    // Canvas and animation variables
    let canvas;
    let animationId;
    let samples = [];
    let weights = [];
    let effectiveSampleSize = 0;
    let totalIterations = 0;
    
    // Sampling parameters
    let proposalDist = 'normal';
    let targetDist = 'bimodal';
    let proposalMean = 0;
    let proposalStdDev = 2;
    
    // Sample range
    const xMin = -10;
    const xMax = 10;
    
    /**
     * Initialize the importance sampling visualization
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
        // Proposal distribution selection
        const proposalSelect = document.getElementById('proposal-dist-is');
        if (proposalSelect) {
            proposalSelect.addEventListener('change', function() {
                proposalDist = this.value;
                drawInitialState();
            });
        }
        
        // Target distribution selection
        const targetSelect = document.getElementById('target-dist-is');
        if (targetSelect) {
            targetSelect.addEventListener('change', function() {
                targetDist = this.value;
                drawInitialState();
            });
        }
        
        // Proposal mean control
        const meanControl = document.getElementById('proposal-mean');
        if (meanControl) {
            meanControl.addEventListener('input', function() {
                proposalMean = parseFloat(this.value);
                document.getElementById('proposal-mean-value').textContent = proposalMean;
                drawInitialState();
            });
        }
        
        // Proposal std dev control
        const stdControl = document.getElementById('proposal-std');
        if (stdControl) {
            stdControl.addEventListener('input', function() {
                proposalStdDev = parseFloat(this.value);
                document.getElementById('proposal-std-value').textContent = proposalStdDev;
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
                weights = [];
                resetStats();
                drawInitialState();
            });
        }
    }
    
    /**
     * Reset statistics counters
     */
    function resetStats() {
        effectiveSampleSize = 0;
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
        
        // Draw target and proposal distributions
        drawDistributions();
        
        // Draw samples histogram if any
        if (samples.length > 0) {
            drawWeightedHistogram();
        }
    }
    
    /**
     * Draw the target and proposal distributions
     */
    function drawDistributions() {
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
        
        // Sample points for drawing distributions
        const points = 200;
        const dx = (xMax - xMin) / points;
        
        // Get target distribution function
        const targetFunction = distributions[targetDist].pdf;
        
        // Get proposal distribution function
        const proposalFunction = (x) => {
            if (proposalDist === 'uniform') {
                // Rescale uniform to cover the range
                return distributions.uniform.pdf(x, xMin, xMax);
            } else if (proposalDist === 'normal') {
                // Use normal with specified parameters
                return distributions.normal.pdf(x, proposalMean, proposalStdDev);
            } else if (proposalDist === 'exponential') {
                // Shift exponential to cover the range
                return distributions.exponential.pdf(x - xMin, 0.5);
            }
            return 0;
        };
        
        // Calculate maximum values to scale
        let maxTargetVal = 0;
        let maxProposalVal = 0;
        
        for (let i = 0; i <= points; i++) {
            const x = xMin + i * dx;
            const targetVal = targetFunction(x);
            const proposalVal = proposalFunction(x);
            
            maxTargetVal = Math.max(maxTargetVal, targetVal);
            maxProposalVal = Math.max(maxProposalVal, proposalVal);
        }
        
        // Scale factor for visualization
        const scaleFactor = Math.max(maxTargetVal, maxProposalVal) > 0 ?
                           1 / Math.max(maxTargetVal, maxProposalVal) :
                           1;
        
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
        
        // Draw proposal distribution
        ctx.beginPath();
        
        for (let i = 0; i <= points; i++) {
            const x = xMin + i * dx;
            const xPixel = padding.left + ((x - xMin) / (xMax - xMin)) * chartWidth;
            const y = proposalFunction(x) * scaleFactor * chartHeight;
            const yPixel = height - padding.bottom - 100 - y;
            
            if (i === 0) {
                ctx.moveTo(xPixel, yPixel);
            } else {
                ctx.lineTo(xPixel, yPixel);
            }
        }
        
        ctx.strokeStyle = '#3498db'; // Proposal distribution color
        ctx.stroke();
        
        // Add legend
        ctx.font = '12px sans-serif';
        
        // Target distribution
        ctx.fillStyle = '#e74c3c';
        ctx.fillRect(padding.left, padding.top, 15, 10);
        ctx.fillStyle = '#333';
        ctx.fillText('Target Distribution', padding.left + 20, padding.top + 10);
        
        // Proposal distribution
        ctx.fillStyle = '#3498db';
        ctx.fillRect(padding.left + 150, padding.top, 15, 10);
        ctx.fillStyle = '#333';
        ctx.fillText('Proposal Distribution', padding.left + 170, padding.top + 10);
    }
    
    /**
     * Draw the weighted histogram of samples
     */
    function drawWeightedHistogram() {
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
        
        // Initialize bins counts (weighted)
        const binCounts = new Array(bins).fill(0);
        const normalizedWeights = normalizeWeights(weights);
        
        // Count samples in each bin with weights
        for (let i = 0; i < samples.length; i++) {
            const sample = samples[i];
            const binIndex = Math.min(Math.floor((sample - xMin) / binWidth), bins - 1);
            if (binIndex >= 0 && binIndex < bins) {
                binCounts[binIndex] += normalizedWeights[i];
            }
        }
        
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
        ctx.fillText('Weighted Histogram of Samples', width / 2, histogramTop + 20);
    }
    
    /**
     * Normalize weights to sum to 1
     */
    function normalizeWeights(weights) {
        if (!weights || weights.length === 0) return [];
        
        const sum = weights.reduce((acc, val) => acc + val, 0);
        
        if (sum === 0) return weights.map(() => 1 / weights.length);
        
        return weights.map(w => w / sum);
    }
    
    /**
     * Calculate effective sample size
     */
    function calculateEffectiveSampleSize(weights) {
        if (!weights || weights.length === 0) return 0;
        
        const normalizedWeights = normalizeWeights(weights);
        const sumSquared = normalizedWeights.reduce((acc, w) => acc + (w * w), 0);
        
        return 1 / sumSquared;
    }
    
    /**
     * Start the importance sampling process
     */
    function startSampling() {
        if (!canvas) return;
        
        // Animation function
        function animate() {
            // Perform a batch of sampling steps
            const batchSize = 10;
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
     * Perform a single importance sampling step
     */
    function performSamplingStep() {
        // Get the target and proposal distribution functions
        const targetFunction = distributions[targetDist].pdf;
        
        // Generate a sample from the proposal distribution
        let proposedSample;
        
        if (proposalDist === 'uniform') {
            proposedSample = distributions.uniform.sample(xMin, xMax);
        } else if (proposalDist === 'normal') {
            proposedSample = distributions.normal.sample(proposalMean, proposalStdDev);
        } else if (proposalDist === 'exponential') {
            proposedSample = distributions.exponential.sample(0.5) + xMin;
        }
        
        // Evaluate proposal density at the sample point
        let proposalDensity;
        
        if (proposalDist === 'uniform') {
            proposalDensity = distributions.uniform.pdf(proposedSample, xMin, xMax);
        } else if (proposalDist === 'normal') {
            proposalDensity = distributions.normal.pdf(proposedSample, proposalMean, proposalStdDev);
        } else if (proposalDist === 'exponential') {
            proposalDensity = distributions.exponential.pdf(proposedSample - xMin, 0.5);
        }
        
        // Evaluate target density at the sample point
        const targetDensity = targetFunction(proposedSample);
        
        // Calculate importance weight
        const weight = proposalDensity > 0 ? targetDensity / proposalDensity : 0;
        
        // Store sample and weight
        samples.push(proposedSample);
        weights.push(weight);
        
        // Update iteration count
        totalIterations++;
        
        // Calculate effective sample size
        effectiveSampleSize = calculateEffectiveSampleSize(weights);
        
        // Update statistics display
        const efficiencyPercent = (effectiveSampleSize / samples.length * 100).toFixed(1);
        document.getElementById('acceptance-rate').textContent = efficiencyPercent + '%';
        document.getElementById('samples-count').textContent = samples.length;
        document.getElementById('iterations-count').textContent = totalIterations;
        
        // Update legend for efficiency
        document.querySelector('.stats-panel h3').textContent = 'Statistics';
        document.querySelector('#acceptance-rate').parentElement.textContent = 
            'Sampling efficiency: ' + document.getElementById('acceptance-rate').outerHTML;
    }
    
    // Export the module
    window.importanceSampling = {
        init: init
    };
})(); 