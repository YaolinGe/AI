/**
 * Utility functions for statistical sampling methods visualization
 */

// Common probability distributions
const distributions = {
    // Standard normal distribution PDF
    normal: {
        pdf: function(x, mean = 0, stdDev = 1) {
            return (1 / (stdDev * Math.sqrt(2 * Math.PI))) * 
                   Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2));
        },
        sample: function(mean = 0, stdDev = 1) {
            // Box-Muller transform
            const u1 = Math.random();
            const u2 = Math.random();
            const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            return z0 * stdDev + mean;
        }
    },

    // Uniform distribution PDF
    uniform: {
        pdf: function(x, min = 0, max = 1) {
            return (x >= min && x <= max) ? 1 / (max - min) : 0;
        },
        sample: function(min = 0, max = 1) {
            return min + Math.random() * (max - min);
        }
    },

    // Exponential distribution PDF
    exponential: {
        pdf: function(x, lambda = 1) {
            return (x >= 0) ? lambda * Math.exp(-lambda * x) : 0;
        },
        sample: function(lambda = 1) {
            return -Math.log(1 - Math.random()) / lambda;
        }
    },

    // Gamma distribution PDF (simplified)
    gamma: {
        pdf: function(x, shape = 1, scale = 1) {
            if (x <= 0) return 0;
            // This is a simplification; real gamma PDF is more complex
            return Math.pow(x, shape - 1) * Math.exp(-x / scale) / 
                   (Math.pow(scale, shape) * gamma(shape));
        }
    },

    // Bimodal distribution (mixture of two normals)
    bimodal: {
        pdf: function(x) {
            return 0.5 * distributions.normal.pdf(x, -2, 1) + 
                   0.5 * distributions.normal.pdf(x, 2, 1);
        },
        sample: function() {
            return Math.random() < 0.5 ? 
                distributions.normal.sample(-2, 1) : 
                distributions.normal.sample(2, 1);
        }
    },

    // Custom target distribution (can be changed to demonstrate different scenarios)
    custom: {
        pdf: function(x) {
            // Example: a mixture of normal and exponential
            return 0.7 * distributions.normal.pdf(x, 1, 2) + 
                   0.3 * distributions.exponential.pdf(x - 5, 0.5);
        }
    }
};

// Math utilities
function gamma(n) {
    // Simple gamma function approximation for integers and half-integers
    if (n === 1) return 1;
    if (n === 0.5) return Math.sqrt(Math.PI);
    return (n - 1) * gamma(n - 1);
}

// Visualization utilities
function createHistogram(canvas, data, bins = 30, color = '#3498db', overlayFunction = null, overlayColor = '#e74c3c') {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // No data case
    if (!data || data.length === 0) {
        ctx.font = '16px sans-serif';
        ctx.fillStyle = '#666';
        ctx.textAlign = 'center';
        ctx.fillText('No data to display', width / 2, height / 2);
        return;
    }
    
    // Calculate histogram bins
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min;
    const binWidth = range / bins;
    
    // Count values in each bin
    const counts = new Array(bins).fill(0);
    
    data.forEach(value => {
        const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
        counts[binIndex]++;
    });
    
    // Find the maximum count to scale the histogram
    const maxCount = Math.max(...counts);
    
    // Padding
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
    
    // Draw histogram bars
    const barWidth = chartWidth / bins;
    
    ctx.fillStyle = color;
    for (let i = 0; i < bins; i++) {
        const barHeight = (counts[i] / maxCount) * chartHeight;
        ctx.fillRect(
            padding.left + i * barWidth,
            height - padding.bottom - barHeight,
            barWidth - 1,
            barHeight
        );
    }
    
    // Draw overlay function if provided
    if (overlayFunction) {
        // Sample the function over the range
        const points = [];
        const samples = 100;
        let maxY = 0;
        
        for (let i = 0; i <= samples; i++) {
            const x = min + (i / samples) * range;
            const y = overlayFunction(x);
            points.push({ x, y });
            maxY = Math.max(maxY, y);
        }
        
        // Normalize the function values to fit in the chart
        const normalizedPoints = points.map(p => ({
            x: padding.left + ((p.x - min) / range) * chartWidth,
            y: height - padding.bottom - (p.y / maxY) * chartHeight * 0.95
        }));
        
        // Draw the function curve
        ctx.beginPath();
        ctx.moveTo(normalizedPoints[0].x, normalizedPoints[0].y);
        
        for (let i = 1; i < normalizedPoints.length; i++) {
            ctx.lineTo(normalizedPoints[i].x, normalizedPoints[i].y);
        }
        
        ctx.strokeStyle = overlayColor;
        ctx.lineWidth = 2;
        ctx.stroke();
    }
    
    // Draw x-axis labels
    ctx.fillStyle = '#333';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    
    for (let i = 0; i <= 5; i++) {
        const x = padding.left + (i / 5) * chartWidth;
        const value = min + (i / 5) * range;
        ctx.fillText(value.toFixed(1), x, height - padding.bottom + 15);
    }
    
    // Draw y-axis label
    ctx.save();
    ctx.translate(padding.left - 25, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Frequency', 0, 0);
    ctx.restore();
}

function createScatterPlot(canvas, data, color = '#3498db', size = 3) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // No data case
    if (!data || data.length === 0) {
        ctx.font = '16px sans-serif';
        ctx.fillStyle = '#666';
        ctx.textAlign = 'center';
        ctx.fillText('No data to display', width / 2, height / 2);
        return;
    }
    
    // Find min and max for both dimensions
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    data.forEach(point => {
        minX = Math.min(minX, point.x);
        maxX = Math.max(maxX, point.x);
        minY = Math.min(minY, point.y);
        maxY = Math.max(maxY, point.y);
    });
    
    // Add some padding
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    minX -= rangeX * 0.05;
    maxX += rangeX * 0.05;
    minY -= rangeY * 0.05;
    maxY += rangeY * 0.05;
    
    // Padding
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
    const mapX = x => padding.left + ((x - minX) / (maxX - minX)) * chartWidth;
    const mapY = y => height - padding.bottom - ((y - minY) / (maxY - minY)) * chartHeight;
    
    // Plot points
    data.forEach(point => {
        ctx.beginPath();
        ctx.arc(
            mapX(point.x),
            mapY(point.y),
            size,
            0,
            Math.PI * 2
        );
        ctx.fillStyle = point.color || color;
        ctx.fill();
    });
    
    // Draw x-axis labels
    ctx.fillStyle = '#333';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    
    for (let i = 0; i <= 5; i++) {
        const x = padding.left + (i / 5) * chartWidth;
        const value = minX + (i / 5) * (maxX - minX);
        ctx.fillText(value.toFixed(1), x, height - padding.bottom + 15);
    }
    
    // Draw y-axis labels
    ctx.textAlign = 'right';
    
    for (let i = 0; i <= 5; i++) {
        const y = height - padding.bottom - (i / 5) * chartHeight;
        const value = minY + (i / 5) * (maxY - minY);
        ctx.fillText(value.toFixed(1), padding.left - 5, y + 5);
    }
}

// Function to animate the sampling process
function animateSampling(canvas, samplingFunc, delay = 50, maxSamples = 1000) {
    let samples = [];
    let count = 0;
    
    const animate = () => {
        const newSamples = samplingFunc();
        samples = samples.concat(newSamples);
        
        // Update visualization
        createHistogram(canvas, samples);
        
        count += newSamples.length;
        
        // Update stats
        document.getElementById('samples-count').textContent = samples.length;
        document.getElementById('iterations-count').textContent = count;
        
        // Continue animation if not reached max samples
        if (samples.length < maxSamples) {
            setTimeout(animate, delay);
        }
    };
    
    animate();
    return samples;
}

// Function to update the explanation and controls for a given sampling method
function updateMethodInfo(method) {
    const titleElement = document.getElementById('method-title');
    const descriptionElement = document.getElementById('method-description');
    const stepsElement = document.getElementById('algorithm-steps');
    const controlPanelElement = document.querySelector('.control-panel');
    
    // Clear previous content
    stepsElement.innerHTML = '';
    controlPanelElement.innerHTML = '';
    
    // Set method information based on selected method
    const methodInfo = {
        rejection: {
            title: 'Rejection Sampling',
            description: 'Rejection sampling is a technique for generating random samples from a distribution when direct sampling is difficult. It works by sampling from a simpler proposal distribution and accepting or rejecting samples based on a comparison with the target distribution.',
            steps: [
                'Define a target distribution p(x) that we want to sample from.',
                'Choose a proposal distribution q(x) that is easy to sample from and covers the target distribution.',
                'Find a constant M such that p(x) ≤ M·q(x) for all x.',
                'Sample a point x from the proposal distribution q(x).',
                'Generate a uniform random number u from [0, 1].',
                'If u ≤ p(x)/(M·q(x)), accept x as a sample from p(x); otherwise, reject it.',
                'Repeat until you have the desired number of samples.'
            ],
            controls: `
                <label for="proposal-dist">Proposal Distribution:</label>
                <select id="proposal-dist">
                    <option value="uniform">Uniform</option>
                    <option value="normal">Normal</option>
                    <option value="exponential">Exponential</option>
                </select>
                
                <label for="target-dist">Target Distribution:</label>
                <select id="target-dist">
                    <option value="bimodal">Bimodal</option>
                    <option value="gamma">Gamma</option>
                    <option value="custom">Custom</option>
                </select>
                
                <label for="scaling-factor">Scaling Factor (M):</label>
                <input type="range" id="scaling-factor" min="1" max="5" step="0.1" value="2">
                <span id="scaling-factor-value">2.0</span>
            `
        },
        importance: {
            title: 'Importance Sampling',
            description: 'Importance sampling is a variance reduction technique that estimates properties of a target distribution by sampling from a different distribution and reweighting the samples. It is useful when direct sampling from the target distribution is difficult.',
            steps: [
                'Define a target distribution p(x) that we want to estimate expectations from.',
                'Choose a proposal distribution q(x) that is easy to sample from and has good coverage of p(x).',
                'Generate samples x₁, x₂, ..., xₙ from the proposal distribution q(x).',
                'Calculate importance weights w(xᵢ) = p(xᵢ)/q(xᵢ) for each sample.',
                'Estimate the expectation of a function f(x) under p(x) using the weighted average of f(xᵢ) with weights w(xᵢ).'
            ],
            controls: `
                <label for="proposal-dist-is">Proposal Distribution:</label>
                <select id="proposal-dist-is">
                    <option value="normal">Normal</option>
                    <option value="uniform">Uniform</option>
                    <option value="exponential">Exponential</option>
                </select>
                
                <label for="target-dist-is">Target Distribution:</label>
                <select id="target-dist-is">
                    <option value="bimodal">Bimodal</option>
                    <option value="gamma">Gamma</option>
                    <option value="custom">Custom</option>
                </select>
                
                <label for="proposal-mean">Proposal Mean:</label>
                <input type="range" id="proposal-mean" min="-5" max="5" step="0.5" value="0">
                <span id="proposal-mean-value">0</span>
                
                <label for="proposal-std">Proposal Std Dev:</label>
                <input type="range" id="proposal-std" min="0.5" max="5" step="0.5" value="2">
                <span id="proposal-std-value">2</span>
            `
        },
        mcmc: {
            title: 'Markov Chain Monte Carlo (MCMC)',
            description: 'Markov Chain Monte Carlo (MCMC) is a class of algorithms for sampling from a probability distribution by constructing a Markov chain that has the desired distribution as its equilibrium distribution. Samples are then taken from this chain.',
            steps: [
                'Define a target distribution p(x) that we want to sample from.',
                'Initialize the chain with a starting point x₀.',
                'For each iteration t:',
                '  - Propose a new point x* based on the current point xₜ.',
                '  - Calculate the acceptance probability α = min(1, p(x*)/p(xₜ)).',
                '  - Generate a uniform random number u from [0, 1].',
                '  - If u ≤ α, set xₜ₊₁ = x*; otherwise, set xₜ₊₁ = xₜ.',
                'After a burn-in period, collect samples from the chain.'
            ],
            controls: `
                <label for="target-dist-mcmc">Target Distribution:</label>
                <select id="target-dist-mcmc">
                    <option value="bimodal">Bimodal</option>
                    <option value="gamma">Gamma</option>
                    <option value="custom">Custom</option>
                </select>
                
                <label for="proposal-width">Proposal Width:</label>
                <input type="range" id="proposal-width" min="0.1" max="3" step="0.1" value="1">
                <span id="proposal-width-value">1.0</span>
                
                <label for="burn-in">Burn-in Period:</label>
                <input type="range" id="burn-in" min="0" max="500" step="50" value="100">
                <span id="burn-in-value">100</span>
            `
        },
        metropolis: {
            title: 'Metropolis-Hastings Algorithm',
            description: 'The Metropolis-Hastings algorithm is a specific MCMC method that generates samples from a probability distribution using a proposal distribution and an acceptance criterion. It can sample from distributions known only up to a normalizing constant.',
            steps: [
                'Define a target distribution p(x) that we want to sample from (known up to a normalizing constant).',
                'Choose a symmetric proposal distribution q(x*|xₜ) (e.g., Gaussian centered at the current point).',
                'Initialize with a starting point x₀.',
                'For each iteration t:',
                '  - Sample a candidate x* from the proposal distribution q(x*|xₜ).',
                '  - Calculate the acceptance ratio α = min(1, p(x*)/p(xₜ)).',
                '  - Generate a uniform random number u from [0, 1].',
                '  - If u ≤ α, accept the candidate: xₜ₊₁ = x*.',
                '  - Otherwise, reject the candidate and stay at the current point: xₜ₊₁ = xₜ.',
                'After a burn-in period, collect samples from the chain.'
            ],
            controls: `
                <label for="target-dist-mh">Target Distribution:</label>
                <select id="target-dist-mh">
                    <option value="bimodal">Bimodal</option>
                    <option value="gamma">Gamma</option>
                    <option value="custom">Custom</option>
                </select>
                
                <label for="proposal-std-mh">Proposal Std Dev:</label>
                <input type="range" id="proposal-std-mh" min="0.1" max="5" step="0.1" value="1">
                <span id="proposal-std-mh-value">1.0</span>
                
                <label for="show-trajectory">Show Trajectory:</label>
                <input type="checkbox" id="show-trajectory" checked>
            `
        },
        gibbs: {
            title: 'Gibbs Sampling',
            description: 'Gibbs sampling is a special case of MCMC that is applicable when the joint distribution is difficult to sample from directly but the conditional distributions of each variable given the others are known and easy to sample from.',
            steps: [
                'Define a target joint distribution p(x₁, x₂, ..., xₙ) that we want to sample from.',
                'Initialize with starting values for all variables x₀ = (x₁₀, x₂₀, ..., xₙ₀).',
                'For each iteration t:',
                '  - For each dimension i from 1 to n:',
                '    * Sample x_i^(t+1) from the conditional distribution p(x_i | x_1^(t+1), ..., x_(i-1)^(t+1), x_(i+1)^(t), ..., x_n^(t)).',
                'After a burn-in period, collect samples from the chain.'
            ],
            controls: `
                <label for="correlation">Correlation Parameter:</label>
                <input type="range" id="correlation" min="-0.9" max="0.9" step="0.1" value="0.5">
                <span id="correlation-value">0.5</span>
                
                <label for="gibbs-steps-per-sample">Steps per Sample:</label>
                <input type="range" id="gibbs-steps-per-sample" min="1" max="10" step="1" value="1">
                <span id="gibbs-steps-value">1</span>
                
                <label for="show-path">Show Path:</label>
                <input type="checkbox" id="show-path" checked>
            `
        }
    };
    
    // Update title and description
    titleElement.textContent = methodInfo[method].title;
    descriptionElement.textContent = methodInfo[method].description;
    
    // Add steps
    methodInfo[method].steps.forEach(step => {
        const stepElement = document.createElement('p');
        stepElement.textContent = step;
        stepsElement.appendChild(stepElement);
    });
    
    // Add controls
    controlPanelElement.innerHTML = methodInfo[method].controls;
    
    // Add event listeners for range inputs
    document.querySelectorAll('input[type="range"]').forEach(input => {
        const valueSpan = document.getElementById(`${input.id}-value`);
        if (valueSpan) {
            valueSpan.textContent = input.value;
            input.addEventListener('input', () => {
                valueSpan.textContent = input.value;
            });
        }
    });
}

// Export utilities
window.utils = {
    distributions,
    createHistogram,
    createScatterPlot,
    animateSampling,
    updateMethodInfo
}; 