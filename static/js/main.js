/**
 * Disease Prediction System - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Disease Prediction System loaded');
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Form validation enhancement
    enhanceForms();
    
    // Add animation to result cards
    animateResults();
    
    // Initialize charts if needed
    initializeCharts();
});

/**
 * Enhance form validation with real-time feedback
 */
function enhanceForms() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input[type="number"], select');
        
        inputs.forEach(input => {
            // Add input validation
            input.addEventListener('blur', function() {
                validateInput(this);
            });
            
            // Add real-time feedback for range inputs
            if (input.type === 'number' && input.min && input.max) {
                input.addEventListener('input', function() {
                    updateRangeIndicator(this);
                });
            }
        });
        
        // Prevent form submission if invalid
        form.addEventListener('submit', function(e) {
            let isValid = true;
            
            inputs.forEach(input => {
                if (!validateInput(input)) {
                    isValid = false;
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                showToast('Please check all fields for errors.', 'error');
            }
        });
    });
}

/**
 * Validate a single input field
 */
function validateInput(input) {
    const value = input.value;
    const min = input.min ? parseFloat(input.min) : null;
    const max = input.max ? parseFloat(input.max) : null;
    
    let isValid = true;
    let message = '';
    
    // Check required
    if (input.required && !value.trim()) {
        isValid = false;
        message = 'This field is required.';
    }
    
    // Check min/max for numbers
    if (input.type === 'number' && value) {
        const numValue = parseFloat(value);
        
        if (min !== null && numValue < min) {
            isValid = false;
            message = `Minimum value is ${min}.`;
        }
        
        if (max !== null && numValue > max) {
            isValid = false;
            message = `Maximum value is ${max}.`;
        }
    }
    
    // Update UI
    const feedbackDiv = input.nextElementSibling;
    if (feedbackDiv && feedbackDiv.classList.contains('form-text')) {
        if (!isValid) {
            feedbackDiv.style.color = '#dc3545';
            feedbackDiv.textContent = message;
            input.classList.add('is-invalid');
        } else {
            feedbackDiv.style.color = '#6c757d';
            input.classList.remove('is-invalid');
            input.classList.add('is-valid');
        }
    }
    
    return isValid;
}

/**
 * Update range indicator for number inputs
 */
function updateRangeIndicator(input) {
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const value = parseFloat(input.value) || min;
    
    // Calculate percentage
    const percent = ((value - min) / (max - min)) * 100;
    
    // Find or create indicator
    let indicator = input.parentElement.querySelector('.range-indicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.className = 'range-indicator mt-2';
        indicator.innerHTML = `
            <div class="d-flex justify-content-between">
                <small>${min}</small>
                <small>Current: ${value}</small>
                <small>${max}</small>
            </div>
            <div class="progress" style="height: 5px;">
                <div class="progress-bar" role="progressbar" style="width: ${percent}%"></div>
            </div>
        `;
        input.parentElement.appendChild(indicator);
    } else {
        indicator.querySelector('.progress-bar').style.width = percent + '%';
        indicator.querySelector('small:nth-child(2)').textContent = `Current: ${value}`;
    }
}

/**
 * Animate result elements
 */
function animateResults() {
    const results = document.querySelectorAll('.alert-result, .result-card');
    
    results.forEach((result, index) => {
        result.style.opacity = '0';
        result.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            result.style.transition = 'opacity 0.5s, transform 0.5s';
            result.style.opacity = '1';
            result.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.id = toastId;
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                    data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    // Show toast
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // Remove toast after hiding
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

/**
 * Initialize charts for visualization
 */
function initializeCharts() {
    // Check if we're on a page that needs charts
    const chartContainers = document.querySelectorAll('.chart-container');
    
    if (chartContainers.length > 0 && typeof Chart !== 'undefined') {
        chartContainers.forEach(container => {
            const chartType = container.dataset.chartType;
            const chartData = JSON.parse(container.dataset.chartData);
            
            if (chartType === 'featureImportance') {
                createFeatureImportanceChart(container, chartData);
            }
        });
    }
}

/**
 * Create feature importance chart
 */
function createFeatureImportanceChart(container, data) {
    const labels = Object.keys(data);
    const values = Object.values(data);
    
    // Sort by value (descending)
    const sortedIndices = values.map((v, i) => i)
        .sort((a, b) => values[b] - values[a]);
    
    const sortedLabels = sortedIndices.map(i => labels[i]);
    const sortedValues = sortedIndices.map(i => values[i]);
    
    const ctx = document.createElement('canvas');
    container.appendChild(ctx);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedLabels,
            datasets: [{
                label: 'Importance',
                data: sortedValues,
                backgroundColor: 'rgba(102, 126, 234, 0.8)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Importance: ${context.raw.toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Importance Score'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Features'
                    }
                }
            }
        }
    });
}

/**
 * Calculate BMI from weight and height
 */
function calculateBMI(weight, height) {
    // Weight in kg, height in meters
    return weight / (height * height);
}

/**
 * Format medical values
 */
function formatMedicalValue(value, unit) {
    if (value === null || value === undefined) return 'N/A';
    return `${parseFloat(value).toFixed(2)} ${unit}`;
}

/**
 * Copy prediction results to clipboard
 */
function copyResultsToClipboard() {
    const resultText = document.querySelector('.result-text');
    if (resultText) {
        navigator.clipboard.writeText(resultText.textContent)
            .then(() => showToast('Results copied to clipboard!', 'success'))
            .catch(() => showToast('Failed to copy results', 'error'));
    }
}

/**
 * Share prediction results
 */
function shareResults() {
    if (navigator.share) {
        const resultText = document.querySelector('.result-text');
        navigator.share({
            title: 'Disease Prediction Results',
            text: resultText ? resultText.textContent : 'Check out my disease prediction results!',
            url: window.location.href
        });
    } else {
        copyResultsToClipboard();
    }
}

/**
 * Load sample data for testing
 */
function loadSampleData(diseaseType) {
    const samples = {
        heart: {
            age: 55,
            sex: 1,
            cp: 0,
            trestbps: 130,
            chol: 250,
            fbs: 0,
            restecg: 1,
            thalach: 150,
            exang: 0,
            oldpeak: 1.0,
            slope: 2,
            ca: 0,
            thal: 2
        },
        diabetes: {
            pregnancies: 3,
            glucose: 120,
            blood_pressure: 70,
            skin_thickness: 20,
            insulin: 79,
            bmi: 32.0,
            diabetes_pedigree: 0.5,
            age: 33
        },
        cancer: {
            mean_radius: 14.0,
            mean_texture: 19.0,
            mean_perimeter: 92.0,
            mean_area: 650,
            mean_smoothness: 0.095
        }
    };
    
    const sample = samples[diseaseType];
    if (!sample) return;
    
    // Fill form with sample data
    for (const [key, value] of Object.entries(sample)) {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) {
            input.value = value;
            // Trigger input event for visual updates
            input.dispatchEvent(new Event('input'));
        }
    }
    
    showToast('Sample data loaded. Click "Predict" to see results.', 'info');
}

/**
 * Export results as PDF (simplified)
 */
function exportResults() {
    const resultContent = document.querySelector('.result-content');
    if (resultContent) {
        const printWindow = window.open('', '_blank');
        printWindow.document.write(`
            <html>
                <head>
                    <title>Disease Prediction Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; padding: 20px; }
                        h1 { color: #333; }
                        .result { font-size: 24px; font-weight: bold; margin: 20px 0; }
                        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                    </style>
                </head>
                <body>
                    <h1>Disease Prediction Report</h1>
                    <div>${resultContent.innerHTML}</div>
                    <p><small>Generated on ${new Date().toLocaleString()}</small></p>
                </body>
            </html>
        `);
        printWindow.document.close();
        printWindow.print();
    }
}

// Make functions available globally for onclick events
window.loadSampleData = loadSampleData;
window.copyResultsToClipboard = copyResultsToClipboard;
window.shareResults = shareResults;
window.exportResults = exportResults;