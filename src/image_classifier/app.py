from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    print("someone hit /health!")
    return {"status": "ok"}

def main():
    print("Hello from image-classifier!")   