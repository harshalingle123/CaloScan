import kagglehub

# Download latest version
path = kagglehub.dataset_download("data/food41")

print("Path to dataset files:", path)