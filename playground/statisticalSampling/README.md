# Statistical Sampling Methods Visualizer

An interactive web-based visualization tool for understanding different statistical sampling methods.

## Overview

This application provides an intuitive, visual way to understand the mechanics and differences between several important sampling techniques used in statistics, machine learning, and Bayesian inference:

- **Rejection Sampling**: Visualize how samples are accepted or rejected based on comparison with a target distribution.
- **Importance Sampling**: See how samples from a proposal distribution are weighted to approximate a target distribution.
- **Markov Chain Monte Carlo (MCMC)**: Observe how the Markov chain traverses the sample space and converges to the target distribution.
- **Metropolis-Hastings Algorithm**: Interact with specific parameters of this MCMC method and visualize the acceptance/rejection process.
- **Gibbs Sampling**: See how this technique samples from conditional distributions in a multivariate setting.

## Features

- Interactive visualizations for each sampling method
- Real-time animation of the sampling process
- Adjustable parameters to explore different scenarios
- Visual comparisons between target and proposal distributions
- Statistics tracking for acceptance rates, sample counts, and more
- Clear explanations of each method's algorithm

## How to Use

1. Open `index.html` in a web browser
2. Select a sampling method from the sidebar
3. Adjust the parameters using the provided controls
4. Click "Run Sampling" to start the visualization
5. Click "Reset" to clear the current samples and start over

## Technical Details

This visualization is built with pure JavaScript and HTML5 Canvas, without any heavy external libraries. The only external dependencies are:

- Chart.js (for some data visualization aspects)
- D3.js (for some advanced visualizations)

## Learning Objectives

This tool is designed to help understand:

- The core principles behind each sampling method
- The strengths and limitations of different approaches
- How parameter choices affect sampling efficiency
- The visual intuition behind statistical sampling algorithms
- The relationship between proposal and target distributions

## Development

To modify or extend this visualization:

- The main application logic is in `js/main.js`
- Each sampling method has its own implementation file in the `js` directory
- Utility functions for distributions and visualization are in `js/utils.js`
- Styles can be modified in `styles.css` 