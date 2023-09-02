document.addEventListener("DOMContentLoaded", function () {
        const form = document.querySelector("form");
        form.addEventListener("submit", function (event) {
            // Check if the skills input is empty
            const skillsInput = document.querySelector('input[name="skills"]').value.trim();
            if (skillsInput === "") {
                alert("Please enter technical skills proficiency  before matching.");
                event.preventDefault(); // Prevent form submission
                return;
            }

            // Check if the resume and JD files are selected
            const resumeInput = document.querySelector('input[name="pdf"]');
            const jdInput = document.querySelector('input[name="jd"]');
            if (!resumeInput.files.length || !jdInput.files.length) {
                alert("Please upload both resume files and job description.");
                event.preventDefault(); // Prevent form submission
                return;
            }

            // If all checks pass, the form will submit and redirect to result.html
        });
    });