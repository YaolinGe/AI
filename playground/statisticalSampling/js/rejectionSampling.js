/**
 * Rejection Sampling Implementation
 */

(function() {
    // Reference to the utility functions
    const utils = window.utils;
    const distributions = utils.distributions;
    
    // Canvas and animation variables
    let canvas;
    let animationId;
    let samples = [];
    let rejectedSamples = [];
    let acceptanceRate = 0;
    let totalIterations = 0;
    
    // Sampling parameters
    let proposalDist = 'uniform';
    let targetDist = 'bimodal';
    let scalingFactor = 2.0;
    
    // Sample range
    const xMin = -10;
    const xMax = 10;
    
    /**
     * Initialize the rejection sampling visualization
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
        const proposalSelect = document.getElementById('proposal-dist');
        if (proposalSelect) {
            proposalSelect.addEventListener('change', function() {
                proposalDist = this.value;
                drawInitialState();
            });
        }
        
        // Target distribution selection
        const targetSelect = document.getElementById('target-dist');
        if (targetSelect) {
            targetSelect.addEventListener('change', function() {
                targetDist = this.value;
                drawInitialState();
            });
        }
        
        // Scaling factor control
        const scalingControl = document.getElementById('scaling-factor');
        if (scalingControl) {
            scalingControl.addEventListener('input', function() {
                scalingFactor = parseFloat(this.value);
                document.getElementById('scaling-factor-value').textContent = scalingFactor.toFixed(1);
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
                rejectedSamples = [];
                resetStats();
                drawInitialState();
            });
        }
    }
    
    /**
     * Reset statistics counters
     */
    function resetStats() {
        acceptanceRate = 0;
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
        
        // Draw samples if any
        if (samples.length > 0) {
            drawSamples();
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
        const chartHeight = height - padding.top - padding.bottom - 100; // Leave space for samples
        
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
                // Use normal centered at 0 with appropriate std dev
                return distributions.normal.pdf(x, 0, 3);
            } else if (proposalDist === 'exponential') {
                // Shift exponential to cover positive and negative values
                return distributions.exponential.pdf(x + 10, 0.5);
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
        const scaleFactor = maxProposalVal * scalingFactor > maxTargetVal ?
                           1 / (maxProposalVal * scalingFactor) :
                           1 / maxTargetVal;
        
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
        
        // Draw proposal distribution (scaled by M)
        ctx.beginPath();
        
        for (let i = 0; i <= points; i++) {
            const x = xMin + i * dx;
            const xPixel = padding.left + ((x - xMin) / (xMax - xMin)) * chartWidth;
            const y = proposalFunction(x) * scalingFactor * scaleFactor * chartHeight;
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
        ctx.fillText('M × Proposal Distribution', padding.left + 170, padding.top + 10);
    }
    
    /**
     * Draw the accepted and rejected samples
     */
    function drawSamples() {
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Parameters for drawing
        const padding = { top: 20, right: 20, bottom: 30, left: 40 };
        const chartWidth = width - padding.left - padding.right;
        
        // Draw rejected samples first (so accepted samples appear on top)
        rejectedSamples.forEach(sample => {
            const xPixel = padding.left + ((sample.x - xMin) / (xMax - xMin)) * chartWidth;
            const yPixel = height - padding.bottom - 50; // Position samples near the bottom
            
            ctx.beginPath();
            ctx.arc(xPixel, yPixel, 3, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(231, 76, 60, 0.3)'; // Transparent red for rejected
            ctx.fill();
        });
        
        // Draw accepted samples
        samples.forEach(sample => {
            const xPixel = padding.left + ((sample - xMin) / (xMax - xMin)) * chartWidth;
            const yPixel = height - padding.bottom - 50; // Position samples near the bottom
            
            ctx.beginPath();
            ctx.arc(xPixel, yPixel, 3, 0, Math.PI * 2);
            ctx.fillStyle = '#2ecc71'; // Green for accepted
            ctx.fill();
        });
    }
    
    /**
     * Start the rejection sampling process
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
     * Perform a single rejection sampling step
     */
    function performSamplingStep() {
        // Get the target and proposal distribution functions
        const targetFunction = distributions[targetDist].pdf;
        
        // Generate a sample from the proposal distribution
        let proposedSample;
        
        if (proposalDist === 'uniform') {
            proposedSample = distributions.uniform.sample(xMin, xMax);
        } else if (proposalDist === 'normal') {
            proposedSample = distributions.normal.sample(0, 3);
        } else if (proposalDist === 'exponential') {
            proposedSample = distributions.exponential.sample(0.5) - 10;
        }
        
        // Evaluate proposal density at the sample point
        let proposalDensity;
        
        if (proposalDist === 'uniform') {
            proposalDensity = distributions.uniform.pdf(proposedSample, xMin, xMax);
        } else if (proposalDist === 'normal') {
            proposalDensity = distributions.normal.pdf(proposedSample, 0, 3);
        } else if (proposalDist === 'exponential') {
            proposalDensity = distributions.exponential.pdf(proposedSample + 10, 0.5);
        }
        
        // Evaluate target density at the sample point
        const targetDensity = targetFunction(proposedSample);
        
        // Calculate acceptance probability
        const acceptanceProb = targetDensity / (scalingFactor * proposalDensity);
        
        // Generate uniform random number for acceptance test
        const u = Math.random();
        
        // Update iteration count
        totalIterations++;
        
        // Accept or reject the sample
        if (u <= acceptanceProb) {
            samples.push(proposedSample);
        } else {
            rejectedSamples.push({ x: proposedSample, y: u * scalingFactor * proposalDensity });
        }
        
        // Update acceptance rate
        acceptanceRate = samples.length / totalIterations;
        
        // Update statistics display
        document.getElementById('acceptance-rate').textContent = (acceptanceRate * 100).toFixed(1) + '%';
        document.getElementById('samples-count').textContent = samples.length;
        document.getElementById('iterations-count').textContent = totalIterations;
    }
    
    // Export the module
    window.rejectionSampling = {
        init: init
    };
})(); 