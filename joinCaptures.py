"""
Este script construye una imagen mosaico de gran tamaño a partir de múltiples
imágenes PNG individuales organizadas en una cuadrícula fija.

FUNCIONAMIENTO GENERAL
----------------------
- El mosaico final está compuesto por 23 filas x 17 columnas de imágenes.
- Cada imagen parcial tiene una resolución fija de 1945 x 920 píxeles.
- Los archivos de entrada siguen la convención de nombres:
      <fila>_<columna>_ahk_test.png
  por ejemplo: 1_1_ahk_test.png, 23_17_ahk_test.png

PROCESO INTERNO
---------------
1. Se calcula la resolución total del mosaico final multiplicando:
      ancho_total  = columnas * ancho_tile
      alto_total   = filas * alto_tile

2. Se crea una única imagen final en memoria (modo RGB), lo cual permite:
   - Pegado incremental de cada imagen parcial
   - Guardado posterior tanto en PNG como en JPG

3. Las imágenes se procesan fila a fila:
   - Cada imagen se abre individualmente
   - Se valida que su tamaño sea el esperado
   - Se pega en su posición exacta dentro del mosaico
   - Se cierra inmediatamente para liberar memoria

4. El resultado se guarda en formato JPG con:
   - Calidad configurable (por defecto 85)
   - Subsampling 4:2:0 para reducir tamaño
   - Optimización del encoder JPEG activada

CONSIDERACIONES IMPORTANTES
---------------------------
- La imagen final es extremadamente grande (decenas de miles de píxeles por lado).
- El consumo de memoria es elevado: se recomienda disponer de al menos 16–32 GB de RAM.
- El script prioriza robustez y control del proceso frente a velocidad.
- No se utilizan técnicas de streaming porque Pillow no permite escritura JPEG incremental.

Este script está pensado para uso técnico/controlado, no para ejecución en sistemas
con recursos limitados.
"""

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
    quality=JPG_QUALITY,  # 90 mayor calidad
    subsampling=2,  # 0  mayor fidelidad visual
    optimize=True
)

print("Proceso terminado correctamente.")
