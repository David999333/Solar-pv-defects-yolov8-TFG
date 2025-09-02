from ultralytics import YOLO
import os
from multiprocessing import freeze_support


def main():
    # Configuración
    yaml_path = r'C:\\Users\\David\\Documents\\David\\TFG\\Datasets\\DATA4.3yolov8\\DATA4.3yolov8\\data.yaml'
    modelo_base = 'yolov8n.pt'
    carpeta_resultados = r'C:\\Users\\David\\Documents\\David\\TFG\\Python\\Resultados10'
    carpeta_dataset = r'C:\\Users\\David\\Documents\\David\\TFG\\Datasets\\DATA4.3yolov8\\DATA4.3yolov8'

    # Eliminar archivo .cache si existe
    cache_path = os.path.join(carpeta_dataset, 'train.cache')
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"[INFO] Archivo cache eliminado: {cache_path}")

    # Entrenamiento
    print("\n[INFO] Iniciando entrenamiento con YOLOv8...")
    model = YOLO(modelo_base)
    model.train(
        data=yaml_path,
        epochs=40,
        imgsz=640,
        batch=8,
        project=carpeta_resultados,
        name='solar_defectos',
        device='cuda')
    print("[INFO] Entrenamiento completado.\n")

    # Validación
    print("[INFO] Validando el modelo...")
    metrics = model.val()
    print(f"mAP@0.5-0.95: {metrics.box.map:.3f}")
    print(f"mAP@0.5: {metrics.box.map50:.3f}")

    # Predicción
    print("[INFO] Realizando predicción...")
    nueva_ruta = os.path.join(carpeta_dataset, 'test', 'images')
    modelo_entrenado = YOLO(os.path.join(carpeta_resultados, 'solar_defectos', 'weights', 'best.pt'))
    resultados = modelo_entrenado.predict(source=nueva_ruta, save=True, save_txt=True, conf=0.25)

    print("\n[INFO] Resultados de predicción:")
    for resultado in resultados:
        for box in resultado.boxes:
            clase = int(box.cls[0])
            confianza = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            print(f"Defecto: {modelo_entrenado.names[clase]} | Confianza: {confianza:.2f} | Caja: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

if __name__ == '__main__':
    freeze_support()
    main()

