import cv2
from ultralytics import YOLO
from deepface import DeepFace

# Cargar modelo YOLOv8 especializado en rostros
model = YOLO("yolov8n-face-lindevs.pt")

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar rostros con YOLOv8
    results = model(frame)

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])

            # Recortar la región de la cara
            face_roi = frame[y1:y2, x1:x2]

            try:
                # Clasificar género con DeepFace
                analysis = DeepFace.analyze(face_roi, actions=['gender'], enforce_detection=False)

                # Extraer género (Male/Female)
                gender = analysis[0]["dominant_gender"]

                # Dibujar rectángulo y etiqueta de género
                color = (255, 0, 0) if gender == "Male" else (255, 20, 147)  # Azul para hombre, rosa para mujer
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            except Exception as e:
                print(f"Error en la clasificación: {e}")

    # Mostrar el video con las detecciones
    cv2.imshow("YOLOv8 - Detección de Rostros y Género", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
