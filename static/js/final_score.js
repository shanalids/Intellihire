function downloadPDF(cand_name, jobrole, highest_matching_percentage, personality_score, ac_score) {
    // Create a new jsPDF instance
    const pdf = new jsPDF();

    // Define the content to be added to the PDF
    const content = `
        The candidate suits the job role they applied for with the following scores:
        Name: ${cand_name}
        Job Role: ${jobrole}
        CV Match: ${highest_matching_percentage}%
        Personality Match: ${personality_score}%
        Academic Transcript Match: ${ac_score}%
    `;

    // Add the content to the PDF
    pdf.text(content, 10, 10);

    // Save the PDF with a specific filename
    pdf.save('final_score.pdf');
    
}
