"""
Simple starter script for CaloScan Web App
This script ensures the upload directory exists before starting the Flask app
"""

import os
import sys

# Create upload directory if it doesn't exist
upload_dir = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(upload_dir, exist_ok=True)

print("=" * 80)
print("CaloScan - Food Nutrition Analyzer")
print("=" * 80)
print(f"\nUpload directory: {upload_dir}")
print("Starting Flask application...\n")

# Import and run the app
from app import app, load_model

if load_model():
    print("\n" + "=" * 80)
    print("SUCCESS: Model loaded successfully!")
    print("=" * 80)
    print("\nOpen your browser and go to:")
    print("   http://localhost:5000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 80 + "\n")

    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
else:
    print("\nERROR: Failed to load model. Please check:")
    print("   1. Model path: E:\\Startup\\CaloScan\\src\\finetuned-food\\checkpoint-3125")
    print("   2. PyTorch and Transformers are installed")
    print("   3. Sufficient GPU/CPU memory available")
    sys.exit(1)
