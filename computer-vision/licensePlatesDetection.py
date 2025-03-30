import yolov5
import cv2

# Cargar el modelo
model = yolov5.load('keremberke/yolov5m-license-plate')

# Establecer parámetros del modelo
model.conf = 0.25  # Umbral de confianza NMS
model.iou = 0.45  # Umbral de IoU NMS
model.agnostic = False  # NMS clase-agnóstica
model.multi_label = False  # NMS múltiples etiquetas por caja
model.max_det = 1000  # número máximo de detecciones por imagen

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

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

    # Guardar resultados en la carpeta "results/"
    results.save(save_dir='results/')

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
