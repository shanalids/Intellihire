const chatbox = document.getElementById('chatbox');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');

sendButton.addEventListener('click', sendMessage);

function sendMessage() {
    const userMessage = userInput.value;
    chatbox.innerHTML += `<p class="user-message">You: ${userMessage}</p>`;

    fetch('/get_bot_response', {
        method: 'POST',
        body: JSON.stringify({ message: userMessage }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        const botResponse = data.bot_response;
        chatbox.innerHTML += `<p class="bot-message">Bot: ${botResponse}</p>`;
    });

    userInput.value = '';



}
