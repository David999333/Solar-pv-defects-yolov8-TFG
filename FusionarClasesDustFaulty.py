from pathlib import Path
import shutil

# === CONFIGURA AQUÍ ===
DATASET_DIR = Path("C:\David\TFG\Datasets\DATA4.3yolov8")  # carpeta que contiene train/, valid/, test/
SPLITS = ["train", "valid", "test"]
MAKE_BACKUPS = True  # crea una copia .bak de cada .txt antes de modificarlo

def process_label_file(txt_path):
    text = txt_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if MAKE_BACKUPS:
        shutil.copy2(txt_path, txt_path.with_suffix(txt_path.suffix + ".bak"))

    converted_2_to_1 = 0
    remapped_3_to_2 = 0

    new_lines = []
    for ln in lines:
        parts = ln.split()
        cls = int(float(parts[0]))  # soporta "0" o "0.0"

        if cls == 2:
            parts[0] = "1"          # dust (2) -> faulty (1)
            converted_2_to_1 += 1
        elif cls == 3:
            parts[0] = "2"          # no_faulty (3) -> (2)
            remapped_3_to_2 += 1

        new_lines.append(" ".join(parts))

    # Guarda (archivo vacío si no hay líneas)
    txt_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    return converted_2_to_1, remapped_3_to_2

def main():
    stats = {"files": 0, "converted_2_to_1": 0, "remapped_3_to_2": 0}

    for split in SPLITS:
        lbl_dir = DATASET_DIR / split / "labels"
        if not lbl_dir.exists():
            continue
        for txt in lbl_dir.rglob("*.txt"):
            c21, r32 = process_label_file(txt)
            stats["files"] += 1
            stats["converted_2_to_1"] += c21
            stats["remapped_3_to_2"] += r32

    print("=== Hecho ===")
    print(f"Archivos procesados:     {stats['files']}")
    print(f"Etiquetas 2→1 convertidas: {stats['converted_2_to_1']}")
    print(f"Etiquetas 3→2 remapeadas:  {stats['remapped_3_to_2']}")

if __name__ == "__main__":
    main()