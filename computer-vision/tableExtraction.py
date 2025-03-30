import os
from ultralyticsplus import YOLO, render_result

# Cargar el modelo
model = YOLO('keremberke/yolov8n-table-extraction')

# Establecer parámetros del modelo
model.overrides['conf'] = 0.25  # Umbral de confianza NMS
model.overrides['iou'] = 0.45  # Umbral de IoU NMS
model.overrides['agnostic_nms'] = False  # NMS clase-agnóstica
model.overrides['max_det'] = 1000  # número máximo de detecciones por imagen

# Ruta de la carpeta que contiene las imágenes
folder_path = '../facturas'  # Cambia esto a la ruta de tu carpeta

# Iterar sobre todos los archivos en la carpeta
for filename in os.listdir(folder_path):
    # Verificar si el archivo es una imagen (puedes agregar más extensiones si es necesario)
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(folder_path, filename)

        # Realizar la inferencia
        results = model.predict(image_path)

        # Observar resultados
        print(f"Resultados para {filename}:")
        print(results[0].boxes)

        # Renderizar y mostrar el resultado
        render = render_result(model=model, image=image_path, result=results[0])
        render.show()
