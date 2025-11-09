# file: mcp_server.py
import pandas as pd, json, pickle, faiss, requests, io
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

# load df / index
# df = pd.read_csv("your.csv")   # or reuse df from above
df = load_dataset("json", data_files={"train":"src/food_instruction_data.jsonl"}, split="train")

with open("src/index_urls.pkl","rb") as f:
    urls = pickle.load(f)
index = faiss.read_index("src/images_index.faiss")

# init CLIP (same as above)
clip_model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_model_id)
clip = CLIPModel.from_pretrained(clip_model_id)
clip = clip.to("cuda" if torch.cuda.is_available() else "cpu")

# Try to import mcp, otherwise create FastAPI endpoints
try:
    fdfd
    from mcp.server import Server
    server = Server("food-nutrition")

    @server.tool()
    def get_nutrition_by_url(image_url: str):
        row = df[df["image_url"] == image_url]
        if not row.empty:
            r = row.iloc[0].to_dict()
            return {
                "dish_name": r.get("dish_name"),
                "ingredients": r.get("ingredients"),
                "food_type": r.get("food_type"),
                "portion_size": r.get("portion_size"),
                "cooking_method": r.get("cooking_method"),
                "nutritional_profile": r.get("nutritional_profile")
            }
        return {"error": "not found"}

    @server.tool()
    def get_by_index(idx: int):
        r = df.iloc[idx].to_dict()
        return r

    @server.tool()
    def search_similar_image(image_bytes: bytes, top_k: int = 3):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(clip.device)
        with torch.no_grad():
            emb = clip.get_image_features(**inputs).cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)
        D, I = index.search(emb, top_k)
        results = []
        for idx in I[0]:
            url = urls[idx]
            row = df[df["image_url"] == url].iloc[0].to_dict()
            results.append(row)
        return results

    if __name__ == "__main__":
        server.run(blocking = True)

except Exception as e:
    # Fallback to FastAPI
    from fastapi import FastAPI, File, UploadFile, Form
    import uvicorn, numpy as np
    app = FastAPI()

    @app.post("/get_nutrition_by_url")
    async def get_nutrition_by_url(image_url: str = Form(...)):
        row = df[df["image_url"] == image_url]
        if not row.empty:
            r = row.iloc[0].to_dict()
            return {
                "dish_name": r.get("dish_name"),
                "ingredients": r.get("ingredients"),
                "food_type": r.get("food_type"),
                "portion_size": r.get("portion_size"),
                "cooking_method": r.get("cooking_method"),
                "nutritional_profile": r.get("nutritional_profile")
            }
        return {"error": "not found"}

    @app.post("/search_similar_image")
    async def search_similar_image(file: UploadFile = File(...), top_k: int = 3):
        b = await file.read()
        img = Image.open(io.BytesIO(b)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(clip.device)
        with torch.no_grad():
            emb = clip.get_image_features(**inputs).cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)
        D, I = index.search(emb, top_k)
        results = []
        for idx in I[0]:
            url = urls[idx]
            row = df[df["image_url"] == url].iloc[0].to_dict()
            results.append(row)
        return {"results": results}

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)
