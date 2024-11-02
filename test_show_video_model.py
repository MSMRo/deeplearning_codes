import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from collections import deque

# Definir el tamaño de la imagen y la longitud de la secuencia
img_height, img_width = 64, 64  # Ajusta estos valores según lo que espera tu modelo
seq_length = 16  # Número de frames por secuencia

# Cargar el modelo de PyTorch
model = torch.load('violence_detection_model.pth')
model.eval()

# Definir las transformaciones
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Inicializar el video
cap = cv2.VideoCapture("V_1.mp4")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Buffer para acumular frames y crear la secuencia
frame_buffer = deque(maxlen=seq_length)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convertir el frame a RGB para Pillow
    pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Preprocesar el frame
    input_tensor = transform(pil_image)
    frame_buffer.append(input_tensor)

    # Procesar cuando el buffer tenga suficientes frames
    if len(frame_buffer) == seq_length:
        # Crear el tensor de entrada con la dimensión de secuencia temporal
        input_batch = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)  # (1, seq_length, C, H, W)

        # Obtener predicción
        with torch.no_grad():
            output = model(input_batch)
            prediction = torch.sigmoid(output)
            prob = prediction.item()
        
        # Mostrar el resultado
        label = "Violence" if prob > 0.5 else "No Violence"
        color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
        
        cv2.putText(frame, f"{label}: {prob:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Violence Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()