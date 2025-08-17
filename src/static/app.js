document.getElementById('alert-button').addEventListener('click', () => {
    const button = document.getElementById('alert-button');
    const loading = document.getElementById('loading');
    const statusMessage = document.getElementById('status-message');
    const resultsContainer = document.getElementById('results');

    // Show loading state
    button.disabled = true;
    button.innerText = 'Detecting...';
    loading.style.display = 'block';
    statusMessage.innerText = '';
    resultsContainer.innerHTML = '';

    // Make the API call to the Flask backend
    fetch('/detect')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                statusMessage.innerText = `Deforestation detected: ${data.percentage}`;
                statusMessage.style.color = '#d32f2f'; // Red for alert

                // Display the images
                const blendedImage = document.createElement('div');
                blendedImage.classList.add('result-image');
                blendedImage.innerHTML = `<h3>Deforestation Overlay</h3><img src="${data.blended_image_url}" alt="Blended Image">`;
                
                const maskImage = document.createElement('div');
                maskImage.classList.add('result-image');
                maskImage.innerHTML = `<h3>Predicted Mask</h3><img src="${data.mask_image_url}" alt="Prediction Mask">`;

                resultsContainer.appendChild(blendedImage);
                resultsContainer.appendChild(maskImage);

            } else {
                statusMessage.innerText = `Error: ${data.error}`;
                statusMessage.style.color = '#333';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            statusMessage.innerText = 'An error occurred. Please check the server logs.';
            statusMessage.style.color = 'red';
        })
        .finally(() => {
            // Revert button state
            button.disabled = false;
            button.innerText = 'Detect Deforestation';
            loading.style.display = 'none';
        });
});