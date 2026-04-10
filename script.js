document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const resultSection = document.getElementById('resultSection');
    const predictBtn = document.getElementById('predictBtn');
    const predictedTimeEl = document.getElementById('predictedTime');
    const statusBadge = document.getElementById('statusBadge');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading state
        const originalBtnText = predictBtn.innerHTML;
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span>Processing...</span><i class="fas fa-spinner fa-spin"></i>';

        // Gather data
        const formData = {
            distance_km: parseFloat(document.getElementById('distance_km').value),
            preparation_time_min: parseFloat(document.getElementById('preparation_time_min').value),
            courier_experience_yrs: parseFloat(document.getElementById('courier_experience_yrs').value),
            weather: document.getElementById('weather').value,
            traffic_level: document.getElementById('traffic_level').value,
            time_of_day: document.getElementById('time_of_day').value,
            vehicle_type: document.querySelector('input[name="vehicle_type"]:checked').value
        };

        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error('Prediction request failed');
            }

            const result = await response.json();

            // Update UI
            predictedTimeEl.textContent = result.predicted_delivery_time;
            statusBadge.textContent = result.delivery_status;
            
            // Handle badge class
            statusBadge.className = 'badge ' + (result.delivery_status === 'On-time' ? 'on-time' : 'late');

            // Show result section
            resultSection.classList.remove('hidden');
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });

        } catch (error) {
            console.error('Error:', error);
            alert('Could not get prediction. Is the backend server running?');
        } finally {
            // Restore button
            predictBtn.disabled = false;
            predictBtn.innerHTML = originalBtnText;
        }
    });
});
