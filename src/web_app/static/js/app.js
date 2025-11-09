// CaloScan - Food Nutrition Analyzer JavaScript

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const uploadForm = document.getElementById('uploadForm');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

// Image upload and preview
fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        // Hide previous results when new image is selected
        resultsSection.style.display = 'none';
        errorMessage.style.display = 'none';

        previewImage(file);
    }
}

function previewImage(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        document.querySelector('.upload-prompt').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Drag and drop functionality
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        // Hide previous results when new image is dropped
        resultsSection.style.display = 'none';
        errorMessage.style.display = 'none';

        fileInput.files = e.dataTransfer.files;
        previewImage(file);
    }
});

// Form submission
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Hide error message
    errorMessage.style.display = 'none';

    // Validate file
    if (!fileInput.files[0]) {
        showError('Please select an image to analyze');
        return;
    }

    // Show loading state
    setLoadingState(true);

    // Create form data
    const formData = new FormData(uploadForm);

    try {
        // Send request to backend
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }

        // Display results
        displayResults(data);

    } catch (error) {
        showError(error.message);
    } finally {
        setLoadingState(false);
    }
});

function setLoadingState(isLoading) {
    const btnText = document.querySelector('.btn-text');
    const btnLoading = document.querySelector('.btn-loading');

    if (isLoading) {
        btnText.style.display = 'none';
        btnLoading.style.display = 'inline-block';
        analyzeBtn.disabled = true;
    } else {
        btnText.style.display = 'inline-block';
        btnLoading.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

function displayResults(data) {
    // Hide upload section and show results
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    // Display the uploaded image
    const resultImage = document.getElementById('resultImage');
    resultImage.src = '/' + data.image_path;

    // Display model used
    const modelUsed = document.getElementById('modelUsed');
    const modelNames = {
        'caloscan-v1': 'CaloScan AI v1 (GPT-2)',
        'effnet-b3': 'Food Classifier (EfficientNet-B3)',
        'finetuned-gpt2': 'CaloScan AI v1 (GPT-2)' // Legacy compatibility
    };
    const modelName = modelNames[data.model_used] || data.model_used;
    modelUsed.innerHTML = `🤖 Model Used: ${modelName}`;

    // Display input summary
    const inputSummary = document.getElementById('inputSummary');
    let summaryHTML = '<h4>📋 Food Information</h4>';

    // For EfficientNet model, show predictions
    if (data.predictions && data.predictions.length > 0) {
        summaryHTML += '<p><strong>Top Predictions:</strong></p><ul>';
        data.predictions.forEach((pred, idx) => {
            summaryHTML += `<li>${pred.class_display} (${(pred.probability * 100).toFixed(1)}%)</li>`;
        });
        summaryHTML += '</ul>';
    }

    // Show standard input data
    if (data.input_data) {
        if (data.input_data.dish_name) {
            summaryHTML += `<p><strong>Dish:</strong> ${data.input_data.dish_name}</p>`;
        }
        if (data.input_data.ingredients) {
            summaryHTML += `<p><strong>Ingredients:</strong> ${data.input_data.ingredients}</p>`;
        }
        if (data.input_data.cooking_method) {
            summaryHTML += `<p><strong>Cooking Method:</strong> ${data.input_data.cooking_method}</p>`;
        }
        if (data.input_data.portion_size) {
            summaryHTML += `<p><strong>Portion Size:</strong> ${data.input_data.portion_size}</p>`;
        }
    }

    inputSummary.innerHTML = summaryHTML;

    // Display nutrition values with animation
    const nutrition = data.nutrition;

    animateValue('calories', nutrition.calories || 0);
    animateValue('protein', nutrition.protein || 0);
    animateValue('fat', nutrition.fat || 0);
    animateValue('carbs', nutrition.carbohydrates || 0);
    animateValue('fiber', nutrition.fiber || 0);

    // Display raw prediction
    document.getElementById('rawPrediction').textContent = data.prediction_text;
}

function animateValue(elementId, finalValue) {
    const element = document.getElementById(elementId);
    const duration = 1000; // 1 second
    const steps = 60;
    const increment = finalValue / steps;
    let current = 0;
    let step = 0;

    const timer = setInterval(() => {
        step++;
        current += increment;

        if (step >= steps) {
            element.textContent = Math.round(finalValue * 10) / 10;
            clearInterval(timer);
        } else {
            element.textContent = Math.round(current * 10) / 10;
        }
    }, duration / steps);
}

function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'block';
    errorMessage.scrollIntoView({ behavior: 'smooth' });
}

function resetForm() {
    // Reset form
    uploadForm.reset();

    // Hide preview
    imagePreview.style.display = 'none';
    document.querySelector('.upload-prompt').style.display = 'block';

    // Hide results
    resultsSection.style.display = 'none';

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Add click event to upload area
uploadArea.addEventListener('click', (e) => {
    if (e.target !== fileInput) {
        fileInput.click();
    }
});

// Prevent form submission on enter key in input fields
document.querySelectorAll('input[type="text"]').forEach(input => {
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
        }
    });
});
