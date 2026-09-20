from fastapi import FastAPI, UploadFile
from PIL import Image

from image_classifier.model import predict

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_image(file: UploadFile):
    image = Image.open(file.file).convert("RGB")
    return predict(image)