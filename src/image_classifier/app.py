from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from image_classifier.model import predict

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_image(file: UploadFile):
    try:
        image = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    return predict(image)