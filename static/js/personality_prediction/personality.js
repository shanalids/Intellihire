// personality.js
// personality.js
document.addEventListener("DOMContentLoaded", function () {
    // Get all range inputs
    const sliders = document.querySelectorAll('input[type="range"]');

    // Function to update slider values
    function updateSliderValue(slider) {
        const sliderValue = slider.parentElement.querySelector('span'); // Find the <span> in the same parent element
        sliderValue.textContent = slider.value; // Update the <span>'s text with the slider's current value
    }

    // Add event listeners to update slider values as they move
    sliders.forEach((slider) => {
        slider.addEventListener('input', () => {
            updateSliderValue(slider);
        });
    });
});


// document.write("<p>This is a simple line printed using JavaScript.</p>");
