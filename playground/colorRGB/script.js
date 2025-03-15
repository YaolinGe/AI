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
    
    // Add event listeners to the sliders
    redSlider.addEventListener('input', updateColor);
    greenSlider.addEventListener('input', updateColor);
    blueSlider.addEventListener('input', updateColor);
    
    // Initialize the color display
    updateColor();
}); 