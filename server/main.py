from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
from torchvision import transforms
import base64
from fastapi.middleware.cors import CORSMiddleware
from src.model.hybrid_model import HybridModel

# Ruta al archivo labels.txt
LABELS_PATH = "/home/lgimenos98/Reconstruction_and_classification/data/food-101/meta/labels.txt" # Cambiar por tu path 

# Ruta al modelo 
MODEL_PATH = "/home/lgimenos98/Reconstruction_and_classification/models/hybrid_vit_base_patch16_224_ep_10.pth" # Cambiar por tu path 

# Cargar las etiquetas
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f]
class_to_label = {i: label for i, label in enumerate(labels)}

# Inicializar FastAPI
app = FastAPI()

# Añadir middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo
model = HybridModel(101, "vit_base_patch16_224") # Elegir el modelo que quieres que se use en el backend de la w
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')), strict=False)
model.eval()

# Preprocesamiento de imagen
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocesar
        input_tensor = preprocess_image(image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_tensor = input_tensor.to(device)

        # Clasificación
        with torch.no_grad():
            _, classification = model(input_tensor)
            predicted_class = torch.argmax(classification, dim=1).item()

        predicted_label = class_to_label[predicted_class]

        # Convertir imagen a Base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        original_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"predicted_label": predicted_label, "original_image": original_image_base64}

    except Exception as e:
        return {"detail": f"Error al procesar la imagen: {str(e)}"}
