import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8 pre-entrenado para detección de rostros
model = YOLO("yolov8n-face-lindevs.pt")  # Usa un modelo entrenado en rostros si es necesario

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar detección de rostrosu5
    results = model(frame)

    # Dibujar los bounding boxes en la imagen
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mostrar el video con detecciones
    cv2.imshow("YOLOv8 Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
