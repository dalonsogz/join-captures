from PIL import Image
import os

# ================= CONFIGURACIÓN =================
ROWS = 23
COLS = 17
TILE_W = 1945
TILE_H = 920

INPUT_DIR = "./captures"
OUTPUT_PNG = "mosaico_final.png"
OUTPUT_JPG = "mosaico_final.jpg"
JPG_QUALITY = 85
# =================================================

final_width = COLS * TILE_W
final_height = ROWS * TILE_H

print(f"Resolución final: {final_width} x {final_height}")

# Crear imagen final (RGB para compatibilidad JPG)
final_img = Image.new("RGB", (final_width, final_height))

for row in range(1, ROWS + 1):
    print(f"Procesando fila {row}/{ROWS}")
    for col in range(1, COLS + 1):
        filename = f"{row}_{col}_ahk_test.png"
        path = os.path.join(INPUT_DIR, filename)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Falta el archivo: {path}")

        with Image.open(path) as tile:
            if tile.size != (TILE_W, TILE_H):
                raise ValueError(f"Tamaño incorrecto en {filename}: {tile.size}")

            x = (col - 1) * TILE_W
            y = (row - 1) * TILE_H
            final_img.paste(tile, (x, y))

# ---------- GUARDADO ----------
# print("Guardando PNG...")
# final_img.save(OUTPUT_PNG, format="PNG", compress_level=6)

print("Guardando JPG...")
final_img.save(
    OUTPUT_JPG,
    format="JPEG",
    quality=JPG_QUALITY,    # 90 mayor calidad
    subsampling=2,  # 0  mayor fidelidad visual
    optimize=True
)

print("Proceso terminado correctamente.")
