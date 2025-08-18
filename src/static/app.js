// static/app.js

document.getElementById('alert-button').addEventListener('click', async () => {
    const button = document.getElementById('alert-button');
    const loading = document.getElementById('loading');
    const statusMessage = document.getElementById('status-message');
    const resultsContainer = document.getElementById('results');
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;

    // Basic validation
    if (!lat || !lon) {
        statusMessage.innerText = 'Please enter both latitude and longitude.';
        statusMessage.style.color = 'red';
        return;
    }

    // Show loading state
    button.disabled = true;
    button.innerText = 'Detecting...';
    loading.style.display = 'block';
    statusMessage.innerText = 'Downloading and processing the satellite image...';
    statusMessage.style.color = '#333';
    resultsContainer.innerHTML = '';

    try {
        const response = await fetch('/detect', {
            method: 'POST', // Correctly set the method to POST
            headers: {
                'Content-Type': 'application/json',
            },
            // Send the data as a JSON string in the request body
            body: JSON.stringify({
                latitude: parseFloat(lat),
                longitude: parseFloat(lon),
            }),
        });

        const data = await response.json();

        if (response.ok) {
            // Check for success in the response data
            if (data.success) {
                statusMessage.innerText = `Deforestation detected: ${data.deforestation_percentage}%`;
                statusMessage.style.color = '#d32f2f'; // Red for alert

                // Display the images
                resultsContainer.innerHTML = `
                    <div class="result-image">
                        <h3>Deforestation Overlay</h3>
                        <img src="${data.blended_image_url}" alt="Blended Image">
                    </div>
                    <div class="result-image">
                        <h3>Predicted Mask</h3>
                        <img src="${data.mask_image_url}" alt="Prediction Mask">
                    </div>
                `;
            } else {
                statusMessage.innerText = `Error: ${data.error}`;
                statusMessage.style.color = '#333';
            }
        } else {
            // Handle HTTP errors (e.g., 500 Server Error)
            statusMessage.innerText = `An error occurred: ${data.error || 'Server error'}`;
            statusMessage.style.color = 'red';
        }

    } catch (error) {
        console.error('Error:', error);
        statusMessage.innerText = `An error occurred: ${error.message}`;
        statusMessage.style.color = 'red';
    } finally {
        // Revert button state
        button.disabled = false;
        button.innerText = 'Detect Deforestation';
        loading.style.display = 'none';
    }
});