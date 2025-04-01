/**
 * Main Application for Statistical Sampling Methods Visualization
 */

(function() {
    // Current selected sampling method
    let currentMethod = 'rejection';
    
    /**
     * Initialize the application
     */
    function init() {
        // Set up method selector
        const methodItems = document.querySelectorAll('.method-selector li');
        
        methodItems.forEach(item => {
            item.addEventListener('click', function() {
                // Remove active class from all items
                methodItems.forEach(i => i.classList.remove('active'));
                
                // Add active class to clicked item
                this.classList.add('active');
                
                // Get the method name from data attribute
                const method = this.getAttribute('data-method');
                
                // Change the method
                changeMethod(method);
            });
        });
        
        // Initialize with the default method (Rejection Sampling)
        changeMethod(currentMethod);
    }
    
    /**
     * Change the current sampling method
     */
    function changeMethod(method) {
        // Update current method
        currentMethod = method;
        
        // Update the method information (description, steps, controls)
        window.utils.updateMethodInfo(method);
        
        // Initialize the selected method visualization
        switch(method) {
            case 'rejection':
                if (window.rejectionSampling) {
                    window.rejectionSampling.init();
                }
                break;
            case 'importance':
                if (window.importanceSampling) {
                    window.importanceSampling.init();
                }
                break;
            case 'mcmc':
                if (window.mcmc) {
                    window.mcmc.init();
                }
                break;
            case 'metropolis':
                if (window.metropolisHastings) {
                    window.metropolisHastings.init();
                }
                break;
            case 'gibbs':
                if (window.gibbsSampling) {
                    window.gibbsSampling.init();
                }
                break;
        }
    }
    
    // Initialize the application when the DOM is loaded
    document.addEventListener('DOMContentLoaded', init);
})(); 