import yolov5
import cv2
import os
import pytesseract

# Configura la ruta de Tesseract si es necesario
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Cambia esta ruta según tu instalación

# Cargar el modelo
model = yolov5.load('keremberke/yolov5m-license-plate')

# Establecer parámetros del modelo
model.conf = 0.25  # Umbral de confianza NMS
model.iou = 0.45  # Umbral de IoU NMS
model.agnostic = False  # NMS clase-agnóstica
model.multi_label = False  # NMS múltiples etiquetas por caja
model.max_det = 1000  # número máximo de detecciones por imagen

# Crear carpeta para guardar resultados si no existe
save_dir = 'results'
os.makedirs(save_dir, exist_ok=True)

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Contador para nombrar las imágenes
image_counter = 0

while True:
    # Leer un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    # Realizar la inferencia
    results = model(frame, size=640)

    # Mostrar los resultados
    results.show()

    # Obtener las predicciones
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # Procesar cada detección
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        # Extraer la región de la matrícula
        license_plate_img = frame[y1:y2, x1:x2]

        # Usar Tesseract para leer el texto de la matrícula
        license_plate_text = pytesseract.image_to_string(license_plate_img, config='--psm 8')
        print(f"Texto de la matrícula detectada: {license_plate_text.strip()}")

    # Guardar el frame con detecciones en la carpeta "results/"
    image_filename = os.path.join(save_dir, f'detection_{image_counter}.jpg')
    cv2.imwrite(image_filename, frame)
    print(f"Imagen guardada: {image_filename}")

    # Incrementar el contador
    image_counter += 1

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
