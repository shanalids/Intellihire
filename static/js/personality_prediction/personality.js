// Wait for the DOM to be fully loaded
document.addEventListener("DOMContentLoaded", function () {
    // Get references to the rows you want to hide
    var selfRatingRow = document.getElementById("selfRatingOpenEnded");
    var openEndedRow = document.getElementById("openEnded");

    // Get a reference to the toggle button
    var toggleButton = document.getElementById("toggleSelfRatingOpenEnded");

    // Function to toggle the visibility of the rows
    function toggleRows() {
        if (selfRatingRow.style.display === "none") {
            selfRatingRow.style.display = "table-row"; // Display the row
            openEndedRow.style.display = "table-row"; // Display the row
        } else {
            selfRatingRow.style.display = "none"; // Hide the row
            openEndedRow.style.display = "none"; // Hide the row
        }
    }

    // Attach a click event listener to the button
    toggleButton.addEventListener("click", toggleRows);
});



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

// function to validate the uploaded csv
function validatecsv() {
    var fileInput = document.getElementById('csv_file');
    var fileName = fileInput.value;
    var allowedExtensions = /(\.csv)$/i;

    if (!allowedExtensions.exec(fileName)) {
        alert('Please select a CSV file.');
        return false;
    }
    return true;
}

// function to show the instructions when hovered on tooltip icon
$(function () {
    $('[data-toggle="tooltip"]').tooltip();
});

