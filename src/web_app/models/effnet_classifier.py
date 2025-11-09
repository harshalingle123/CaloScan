"""
EfficientNet-B3 Food Classifier
Loads and runs inference with the fine-tuned EfficientNet-B3 model
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import requests
import json

# Food-101 class names
FOOD_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles"
]

class EfficientNetClassifier:
    def __init__(self, model_path, device=None):
        """
        Initialize the EfficientNet-B3 classifier

        Args:
            model_path: Path to the TorchScript model file
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.model_path = model_path

    def load_model(self):
        """Load the TorchScript model"""
        try:
            print(f"Loading EfficientNet-B3 model from {self.model_path}...")
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()

            # Define the same transforms used during training
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            print("EfficientNet-B3 model loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading EfficientNet-B3 model: {e}")
            return False

    def predict(self, image_path, top_k=5):
        """
        Predict food class from image

        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return

        Returns:
            List of tuples (class_name, probability)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                logits = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(logits, dim=1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities[0], k=top_k)

            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                class_name = FOOD_CLASSES[idx.item()]
                predictions.append({
                    'class': class_name,
                    'class_display': class_name.replace('_', ' ').title(),
                    'probability': float(prob.item())
                })

            return predictions

        except Exception as e:
            print(f"Error during prediction: {e}")
            return []


def get_nutrition_from_usda(food_name, api_key, portion_g=100):
    """
    Get nutritional information from USDA FoodData Central

    Args:
        food_name: Name of the food item
        api_key: USDA API key
        portion_g: Portion size in grams (default 100g)

    Returns:
        Dictionary with nutrition information
    """
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        # Return estimated values if no API key
        return get_estimated_nutrition(food_name, portion_g)

    try:
        # Search for food in USDA database
        search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        params = {
            'api_key': api_key,
            'query': food_name,
            'pageSize': 5
        }

        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'foods' in data and len(data['foods']) > 0:
            # Get first result
            food = data['foods'][0]
            nutrients = {}

            # Extract key nutrients
            for nutrient in food.get('foodNutrients', []):
                nutrient_name = nutrient.get('nutrientName', '').lower()
                nutrient_value = nutrient.get('value', 0)

                if 'energy' in nutrient_name and 'kcal' in nutrient_name.lower():
                    nutrients['calories'] = nutrient_value
                elif 'protein' in nutrient_name:
                    nutrients['protein'] = nutrient_value
                elif 'total lipid' in nutrient_name or 'fat' in nutrient_name:
                    nutrients['fat'] = nutrient_value
                elif 'carbohydrate' in nutrient_name:
                    nutrients['carbohydrates'] = nutrient_value
                elif 'fiber' in nutrient_name:
                    nutrients['fiber'] = nutrient_value

            # Scale to portion size (USDA values are per 100g)
            scale_factor = portion_g / 100.0

            return {
                'calories': nutrients.get('calories', 0) * scale_factor,
                'protein': nutrients.get('protein', 0) * scale_factor,
                'fat': nutrients.get('fat', 0) * scale_factor,
                'carbohydrates': nutrients.get('carbohydrates', 0) * scale_factor,
                'fiber': nutrients.get('fiber', 0) * scale_factor,
                'source': 'USDA FoodData Central',
                'food_description': food.get('description', food_name)
            }

    except Exception as e:
        print(f"Error fetching USDA data: {e}")

    # Fallback to estimated values
    return get_estimated_nutrition(food_name, portion_g)


def get_estimated_nutrition(food_name, portion_g=100):
    """
    Get estimated nutritional values based on food category
    This is a fallback when USDA API is unavailable
    """
    # Simplified nutrition estimates per 100g
    nutrition_estimates = {
        # Desserts & Sweets
        'cake': {'calories': 350, 'protein': 4, 'fat': 15, 'carbohydrates': 50, 'fiber': 1},
        'pie': {'calories': 300, 'protein': 3, 'fat': 12, 'carbohydrates': 45, 'fiber': 2},
        'ice_cream': {'calories': 200, 'protein': 4, 'fat': 11, 'carbohydrates': 23, 'fiber': 0.5},
        'donut': {'calories': 450, 'protein': 5, 'fat': 25, 'carbohydrates': 50, 'fiber': 2},
        'chocolate': {'calories': 550, 'protein': 5, 'fat': 35, 'carbohydrates': 55, 'fiber': 3},

        # Protein dishes
        'chicken': {'calories': 165, 'protein': 31, 'fat': 3.6, 'carbohydrates': 0, 'fiber': 0},
        'beef': {'calories': 250, 'protein': 26, 'fat': 15, 'carbohydrates': 0, 'fiber': 0},
        'fish': {'calories': 150, 'protein': 25, 'fat': 5, 'carbohydrates': 0, 'fiber': 0},
        'steak': {'calories': 270, 'protein': 25, 'fat': 19, 'carbohydrates': 0, 'fiber': 0},
        'pork': {'calories': 240, 'protein': 27, 'fat': 14, 'carbohydrates': 0, 'fiber': 0},

        # Carbs
        'rice': {'calories': 130, 'protein': 2.7, 'fat': 0.3, 'carbohydrates': 28, 'fiber': 0.4},
        'pasta': {'calories': 150, 'protein': 5, 'fat': 1, 'carbohydrates': 30, 'fiber': 2},
        'bread': {'calories': 265, 'protein': 9, 'fat': 3.2, 'carbohydrates': 49, 'fiber': 2.7},
        'pizza': {'calories': 266, 'protein': 11, 'fat': 10, 'carbohydrates': 33, 'fiber': 2.5},

        # Salads & Vegetables
        'salad': {'calories': 50, 'protein': 2, 'fat': 1, 'carbohydrates': 8, 'fiber': 3},

        # Default
        'default': {'calories': 150, 'protein': 8, 'fat': 6, 'carbohydrates': 18, 'fiber': 2}
    }

    # Find matching category
    food_lower = food_name.lower().replace('_', ' ')
    nutrition = nutrition_estimates['default'].copy()

    for category, values in nutrition_estimates.items():
        if category in food_lower:
            nutrition = values.copy()
            break

    # Scale to portion size
    scale_factor = portion_g / 100.0

    return {
        'calories': nutrition['calories'] * scale_factor,
        'protein': nutrition['protein'] * scale_factor,
        'fat': nutrition['fat'] * scale_factor,
        'carbohydrates': nutrition['carbohydrates'] * scale_factor,
        'fiber': nutrition['fiber'] * scale_factor,
        'source': 'Estimated',
        'food_description': food_name.replace('_', ' ').title()
    }
