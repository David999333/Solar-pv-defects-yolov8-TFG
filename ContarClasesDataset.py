from pathlib import Path
from collections import defaultdict

# === CONFIGURA AQUÍ ===
DATASET_DIR = Path("C:\David\TFG\Datasets\DATA4.2yolov8")  # carpeta que contiene train/ valid/ test/
SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# Mapea los IDs de clase (los que aparecen como primer término en cada línea del .txt)
ID2NAME = {
    0: "crack",      
    1: "faulty",
    2: "dust",
    3: "no faulty",     
}

# === LÓGICA ===
images_per_class = defaultdict(int)              # imágenes que contienen la clase (no instancias)
instances_per_class = defaultdict(int)           # total de instancias por clase
images_total_per_split = defaultdict(int)        # nº de imágenes por split
images_without_label_per_split = defaultdict(int)

for split in SPLITS:
    img_dir = DATASET_DIR / split / "images"
    lbl_dir = DATASET_DIR / split / "labels"

    # Todas las imágenes del split
    imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    images_total_per_split[split] = len(imgs)

    for img_path in imgs:
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            # Imagen sin anotaciones (negativa). Cuenta si te interesa monitorizarlo.
            images_without_label_per_split[split] += 1
            continue

        # Para "imágenes por clase", hay que sumar 1 por clase distinta presente en ESA imagen
        classes_in_image = set()
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(float(parts[0]))  # por si viene como "0" o "0.0"
                classes_in_image.add(cls_id)
                instances_per_class[cls_id] += 1

        for cls_id in classes_in_image:
            images_per_class[cls_id] += 1

# === IMPRESIÓN DE RESULTADOS ===
print("=== Imágenes totales por split ===")
for s in SPLITS:
    print(f"{s}: {images_total_per_split[s]} (sin label: {images_without_label_per_split[s]})")

print("\n=== Imágenes por clase (cada imagen cuenta una vez por clase) ===")
for cls_id, name in ID2NAME.items():
    print(f"{name}: {images_per_class.get(cls_id, 0)}")

print("\n=== Instancias por clase (suma de líneas en labels) ===")
for cls_id, name in ID2NAME.items():
    print(f"{name}: {instances_per_class.get(cls_id, 0)}")