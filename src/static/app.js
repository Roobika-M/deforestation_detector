// src/static/app.js

/**
 * Main script for the ForestGuard Dashboard.
 * This file handles the logic for the live deforestation detection page.
 */

document.addEventListener('DOMContentLoaded', () => {

    // Get all the necessary DOM elements
    const alertButton = document.getElementById('alert-button');
    const loadingIndicator = document.getElementById('loading');
    const statusMessage = document.getElementById('status-message');
    const resultsContainer = document.getElementById('results');
    const latInput = document.getElementById('lat');
    const lonInput = document.getElementById('lon');
    const dateInput = document.getElementById('date'); // Added date input

    // Listen for the button click event
    if (alertButton) {
        alertButton.addEventListener('click', async () => {
            const lat = latInput.value;
            const lon = lonInput.value;
            const date = dateInput.value;

            // Clear previous results and messages
            statusMessage.innerText = '';
            resultsContainer.innerHTML = '';

            // Check if required fields are filled
            if (!lat || !lon || !date) {
                statusMessage.innerText = 'Error: Missing latitude, longitude, or date.';
                statusMessage.style.color = 'red';
                return;
            }

            // Show loading state
            alertButton.disabled = true;
            alertButton.innerText = 'Detecting...';
            loadingIndicator.style.display = 'block';
            statusMessage.innerText = 'Downloading and processing the satellite image...';
            statusMessage.style.color = 'black';

            try {
                // Construct the API call payload
                const payload = {
                    lat: parseFloat(lat),
                    lon: parseFloat(lon),
                    date: date,
                };

                // Make the API call to the Flask backend
                const response = await fetch('/api/detect-live', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(payload),
                });

                const data = await response.json();

                // Handle the response from the server
                if (response.ok && !data.error) {
                    // Update status message based on deforestation percentage
                    const percentage = data.deforestation_percent;
                    if (percentage > 0.1) {
                        statusMessage.innerText = `🚨 ALERT! Deforestation detected: ${percentage.toFixed(2)}%!`;
                        statusMessage.style.color = '#d32f2f'; // Red for alert
                    } else {
                        statusMessage.innerText = `✅ No significant deforestation detected.`;
                        statusMessage.style.color = '#38a169'; // Green for no alert
                    }
                    
                    // Display the images if they exist
                    if (data.blended_image && data.mask_image) {
                        // Display blended image (original with overlay)
                        const blendedImage = document.createElement('div');
                        blendedImage.classList.add('result-image');
                        blendedImage.innerHTML = `<h3>Deforestation Overlay</h3><img src="data:image/png;base64,${data.blended_image}" alt="Blended Image">`;

                        // Display the mask image
                        const maskImage = document.createElement('div');
                        maskImage.classList.add('result-image');
                        maskImage.innerHTML = `<h3>Predicted Mask</h3><img src="data:image/png;base64,${data.mask_image}" alt="Prediction Mask">`;

                        resultsContainer.appendChild(blendedImage);
                        resultsContainer.appendChild(maskImage);
                    } else {
                         // Handle case where images are not returned
                        statusMessage.innerText = 'Detection was successful, but image data was not returned.';
                        statusMessage.style.color = 'orange';
                    }

                } else {
                    // Handle server-side errors
                    const errorMessage = data.error || 'An unknown error occurred.';
                    statusMessage.innerText = `Error: ${errorMessage}`;
                    statusMessage.style.color = 'red';
                }

            } catch (e) {
                // Handle network or other errors
                console.error('Fetch error:', e);
                statusMessage.innerText = 'Network Error: Could not connect to the server.';
                statusMessage.style.color = 'red';
            } finally {
                // Revert button state and hide loading indicator
                alertButton.disabled = false;
                alertButton.innerText = 'Check Live Deforestation';
                loadingIndicator.style.display = 'none';
            }
        });
    } else {
        console.error("Could not find element with id 'alert-button'.");
    }
});
