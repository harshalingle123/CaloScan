import kagglehub
import os
import shutil

# Download Food101 dataset
print("Downloading Food101 dataset...")
path = kagglehub.dataset_download("data/food101")

# Move it into ./data/food101 folder in your project
target_dir = os.path.join(os.path.dirname(__file__), "..", "data", "food101")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print("Moving dataset to:", target_dir)
shutil.move(path, target_dir)

print("✅ Dataset ready at:", target_dir)
