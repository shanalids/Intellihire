var skills = [];
  var frequencies = [];
  var skillElements = document.querySelectorAll('p');
  skillElements.forEach(function(element) {
    var text = element.textContent.trim().split(' ');
    var skill = text[0];
    
    // Exclude "Technical" and "Keywords" from the x-axis labels
    if (skill !== 'Technical' && skill !== 'Keywords') {
      skills.push(skill);
      frequencies.push(parseInt(text[1]));
    }
  });

  // Define an array of different colors
  var colors = [
    'rgba(75, 192, 192, 0.8)',
    'rgba(255, 99, 132, 0.8)',
    'rgba(54, 162, 235, 0.8)',
    'rgba(255, 206, 86, 0.8)',
    // Add more colors as needed
  ];

  var proficiencyData = {
    labels: skills,
    datasets: [{
      label: 'Skill Frequency',
      data: frequencies,
      backgroundColor: colors,
      borderColor: colors.map(color => color.replace('0.2', '1')), // Use the same colors for border
      borderWidth: 1
    }]
  };

  var proficiencyCtx = document.getElementById('proficiencyChart').getContext('2d');
  var proficiencyChart = new Chart(proficiencyCtx, {
    type: 'bar',
    data: proficiencyData,
    options: {
      scales: {
        y: {
          beginAtZero: true
        }
      }
    }
  });