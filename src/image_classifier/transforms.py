from PIL import Image
from torchvision import transforms


def pad_to_square(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("L", (side, side), color=255)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


symbol_transform = transforms.Compose(
    [
        transforms.Lambda(pad_to_square),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)