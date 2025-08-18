document.getElementById('alert-button').addEventListener('click', () => {
    const button = document.getElementById('alert-button');
    const loading = document.getElementById('loading');
    const statusMessage = document.getElementById('status-message');
    const resultsContainer = document.getElementById('results');
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;

    // Show loading state
    button.disabled = true;
    button.innerText = 'Detecting...';
    loading.style.display = 'block';
    statusMessage.innerText = 'Downloading and processing the satellite image...';
    resultsContainer.innerHTML = '';

    // Make the API call to the Flask backend with lat/lon parameters
    fetch(`/detect?lat=${lat}&lon=${lon}`)
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Server error'); });
            }
            return response.json();
        })
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
                statusMessage.innerText = `An error occurred: ${data.error}`;
                statusMessage.style.color = 'red';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            statusMessage.innerText = `An error occurred: ${error.message}`;
            statusMessage.style.color = 'red';
        })
        .finally(() => {
            // Revert button state
            button.disabled = false;
            button.innerText = 'Detect Deforestation';
            loading.style.display = 'none';
        });
});
