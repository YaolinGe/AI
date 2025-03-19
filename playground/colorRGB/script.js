document.addEventListener('DOMContentLoaded', function() {
    // Get all the elements we need to work with
    const redSlider = document.getElementById('red');
    const greenSlider = document.getElementById('green');
    const blueSlider = document.getElementById('blue');
    
    const redValue = document.getElementById('red-value');
    const greenValue = document.getElementById('green-value');
    const blueValue = document.getElementById('blue-value');
    
    const colorPreview = document.getElementById('color-preview');
    const rgbValue = document.getElementById('rgb-value');
    const hexValue = document.getElementById('hex-value');
    const hexInput = document.getElementById('hex-input');
    
    const redPreview = document.getElementById('red-preview');
    const greenPreview = document.getElementById('green-preview');
    const bluePreview = document.getElementById('blue-preview');
    
    // Function to update the color display
    function updateColor() {
        const red = redSlider.value;
        const green = greenSlider.value;
        const blue = blueSlider.value;
        
        // Update the value displays
        redValue.textContent = red;
        greenValue.textContent = green;
        blueValue.textContent = blue;
        
        // Update the main color preview
        const rgbColor = `rgb(${red}, ${green}, ${blue})`;
        colorPreview.style.backgroundColor = rgbColor;
        rgbValue.textContent = rgbColor;
        
        // Update the hex value
        const hexColor = rgbToHex(red, green, blue);
        hexValue.textContent = hexColor;
        hexInput.value = hexColor;
        
        // Update individual component previews
        redPreview.style.backgroundColor = `rgb(${red}, 0, 0)`;
        greenPreview.style.backgroundColor = `rgb(0, ${green}, 0)`;
        bluePreview.style.backgroundColor = `rgb(0, 0, ${blue})`;
    }
    
    // Function to convert RGB to HEX
    function rgbToHex(r, g, b) {
        return '#' + componentToHex(r) + componentToHex(g) + componentToHex(b);
    }
    
    function componentToHex(c) {
        const hex = parseInt(c).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }
    
    // Function to convert HEX to RGB
    function hexToRgb(hex) {
        // Remove the # if present
        hex = hex.replace(/^#/, '');
        
        // Parse the hex value
        let r, g, b;
        
        if (hex.length === 3) {
            // Short form #RGB
            r = parseInt(hex.charAt(0) + hex.charAt(0), 16);
            g = parseInt(hex.charAt(1) + hex.charAt(1), 16);
            b = parseInt(hex.charAt(2) + hex.charAt(2), 16);
        } else if (hex.length === 6) {
            // Standard form #RRGGBB
            r = parseInt(hex.substring(0, 2), 16);
            g = parseInt(hex.substring(2, 4), 16);
            b = parseInt(hex.substring(4, 6), 16);
        } else {
            // Invalid format
            return null;
        }
        
        return { r, g, b };
    }
    
    // Function to update sliders from hex input
    function updateFromHex() {
        const hex = hexInput.value;
        const rgb = hexToRgb(hex);
        
        if (rgb) {
            // Update sliders
            redSlider.value = rgb.r;
            greenSlider.value = rgb.g;
            blueSlider.value = rgb.b;
            
            // Update the color display
            updateColor();
        }
    }
    
    // Add event listeners to the sliders
    redSlider.addEventListener('input', updateColor);
    greenSlider.addEventListener('input', updateColor);
    blueSlider.addEventListener('input', updateColor);
    
    // Add event listener to hex input
    hexInput.addEventListener('input', function() {
        // Add # if missing
        if (hexInput.value.charAt(0) !== '#') {
            hexInput.value = '#' + hexInput.value;
        }
    });
    
    hexInput.addEventListener('change', updateFromHex);
    
    // Initialize the color display
    updateColor();
}); 