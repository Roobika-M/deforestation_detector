// static/app.js

document.addEventListener('DOMContentLoaded', () => {

    const detectButton = document.getElementById('alert-button');
    const latInput = document.getElementById('lat');
    const lonInput = document.getElementById('lon');
    const loadingSpinner = document.getElementById('loading');
    const statusMessage = document.getElementById('status-message');
    const resultsContainer = document.getElementById('results');

    // --- Map Initialization ---
    // Make sure your HTML has a <div id="map"></div>
    const map = L.map('map').setView([0, 0], 2); // Set initial view to a general location

    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // --- New: Handle Map Clicks ---
    let marker; // To hold the marker for the selected point

    map.on('click', async (e) => {
        const { lat, lng } = e.latlng;
        
        // Remove existing marker if any
        if (marker) {
            map.removeLayer(marker);
        }

        // Add a new marker at the clicked location
        marker = L.marker([lat, lng]).addTo(map);

        // Update the input fields (for user's reference)
        latInput.value = lat.toFixed(6);
        lonInput.value = lng.toFixed(6);

        // Immediately trigger the detection process using the clicked coordinates
        await runDetection(lat, lng);
    });
    
    // --- Original: Handle Button Clicks ---
    // If the user wants to keep the input fields and button, this handles that
    detectButton.addEventListener('click', async () => {
        const lat = latInput.value;
        const lon = lonInput.value;
        await runDetection(lat, lon);
    });

    // --- Core Detection Logic Function ---
    async function runDetection(lat, lon) {
        // Basic validation
        if (!lat || !lon) {
            statusMessage.innerText = 'Please enter both latitude and longitude.';
            statusMessage.style.color = 'red';
            return;
        }

        // Show loading state
        detectButton.disabled = true;
        detectButton.innerText = 'Detecting...';
        loadingSpinner.style.display = 'block';
        statusMessage.innerText = 'Downloading and processing the satellite image...';
        statusMessage.style.color = '#333';
        resultsContainer.innerHTML = '';

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    latitude: parseFloat(lat),
                    longitude: parseFloat(lon),
                }),
            });

            const data = await response.json();

            if (response.ok && data.success) {
                statusMessage.innerText = `Deforestation detected: ${data.deforestation_percentage}%`;
                statusMessage.style.color = '#d32f2f';

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
                statusMessage.innerText = `An error occurred: ${data.error}`;
                statusMessage.style.color = 'red';
            }
        } catch (error) {
            console.error('Fetch error:', error);
            statusMessage.innerText = `An error occurred: ${error.message}`;
            statusMessage.style.color = 'red';
        } finally {
            // Revert button state
            detectButton.disabled = false;
            detectButton.innerText = 'Detect Deforestation';
            loadingSpinner.style.display = 'none';
        }
    }
});
