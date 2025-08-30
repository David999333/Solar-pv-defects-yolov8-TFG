import os
import cv2
import albumentations as A
from glob import glob
from tqdm import tqdm
import numpy as np

# Configuración
CLASES_OBJETIVO = {0: 2500, 3: 1500}  # clase: cantidad actual
MAX_CLASE = 3500  # objetivo para todas
AUMENTOS_POR_IMAGEN = 3

# Rutas
ruta_base = r'C:\David\TFG\Datasets\DATA4.2yolov8\train'
ruta_img = os.path.join(ruta_base, 'images')
ruta_lbl = os.path.join(ruta_base, 'labels')
output_img = os.path.join(ruta_base, 'aug_images')
output_lbl = os.path.join(ruta_base, 'aug_labels')
os.makedirs(output_img, exist_ok=True)
os.makedirs(output_lbl, exist_ok=True)

# Transformaciones
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=25, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.GaussianBlur(p=0.3),
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['class_labels'],
    min_visibility=0.1,
    check_each_transform=True
))

def clamp(value, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, value))

def yolo_to_voc(x, y, w, h, img_w, img_h):
    x_min = (x - w / 2) * img_w
    y_min = (y - h / 2) * img_h
    x_max = (x + w / 2) * img_w
    y_max = (y + h / 2) * img_h
    return [clamp(x_min, 0, img_w), clamp(y_min, 0, img_h), clamp(x_max, 0, img_w), clamp(y_max, 0, img_h)]

def voc_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    x = ((x_min + x_max) / 2) / img_w
    y = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return [clamp(x), clamp(y), clamp(w), clamp(h)]

def is_valid_bbox(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    if x_min >= x_max or y_min >= y_max:
        return False
    if x_min < 0 or x_max > img_w or y_min < 0 or y_max > img_h:
        return False
    return True

# Recuento y agrupación por clase objetivo
archivos = glob(os.path.join(ruta_lbl, '*.txt'))
imagenes_por_clase = {cls: [] for cls in CLASES_OBJETIVO.keys()}
conteo_real = {cls: 0 for cls in CLASES_OBJETIVO.keys()}

for txt in archivos:
    with open(txt, 'r') as f:
        lines = f.readlines()
        clases = [int(l.split()[0]) for l in lines if l.strip()]
        for cls in CLASES_OBJETIVO.keys():
            if cls in clases:
                imagenes_por_clase[cls].append(txt)
                conteo_real[cls] += clases.count(cls)

# Mostrar info
for cls in CLASES_OBJETIVO:
    faltan = MAX_CLASE - conteo_real[cls]
    print(f"[INFO] Clase {cls}: actuales {conteo_real[cls]} → faltan {faltan}")

# Augmentación por clase
for cls_objetivo, lista_txt in imagenes_por_clase.items():
    generadas = 0
    necesarias = MAX_CLASE - conteo_real[cls_objetivo]

    for txt_path in tqdm(lista_txt, desc=f"Aumentando clase {cls_objetivo}"):
        if generadas >= necesarias:
            break

        img_path = txt_path.replace('labels', 'images').replace('.txt', '.jpg')
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] No se pudo cargar la imagen: {img_path}")
            continue
        h_img, w_img = image.shape[:2]

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        bboxes_voc, class_labels = [], []
        for line in lines:
            try:
                cls, x, y, w, h = map(float, line.strip().split())
                voc_box = yolo_to_voc(x, y, w, h, w_img, h_img)
                if is_valid_bbox(voc_box, w_img, h_img):
                    bboxes_voc.append(voc_box)
                    class_labels.append(int(cls))
            except:
                continue

        if not bboxes_voc:
            continue

        for i in range(AUMENTOS_POR_IMAGEN):
            if generadas >= necesarias:
                break

            try:
                augmented = transform(image=image, bboxes=bboxes_voc, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']

                if not aug_bboxes or cls_objetivo not in aug_labels:
                    continue

                final_bboxes = []
                for box in aug_bboxes:
                    yolo_box = voc_to_yolo(*box, aug_img.shape[1], aug_img.shape[0])
                    yolo_box = [round(clamp(b), 6) for b in yolo_box]
                    final_bboxes.append(yolo_box)

                base_name = os.path.basename(txt_path).replace('.txt', f'_cls{cls_objetivo}_aug{generadas}')
                new_img_path = os.path.join(output_img, f'{base_name}.jpg')
                new_txt_path = os.path.join(output_lbl, f'{base_name}.txt')
                cv2.imwrite(new_img_path, aug_img)

                with open(new_txt_path, 'w') as f:
                    for box, cls in zip(final_bboxes, aug_labels):
                        f.write(f"{cls} {' '.join(map(lambda x: f'{x:.6f}', box))}\n")

                generadas += 1

            except Exception as e:
                print(f"[ERROR] {img_path} → {e}")
                continue

    print(f"\n✅ [FINALIZADO] {generadas} imágenes aumentadas para clase {cls_objetivo}.\n")