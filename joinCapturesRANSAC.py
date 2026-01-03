"""
Este script implementa un sistema completo de cosido (“stitching”) de imágenes
basado en estimación automática de desplazamientos mediante visión artificial.

El objetivo es reconstruir una imagen final grande a partir de múltiples capturas
organizadas inicialmente en filas y columnas, donde existe solape parcial entre
imágenes adyacentes tanto en horizontal como en vertical.

────────────────────────────────────────────────────────────
VISIÓN GENERAL DEL PROCESO
────────────────────────────────────────────────────────────

El pipeline se divide en tres fases principales:

1) Estimación de desplazamientos entre imágenes consecutivas
2) Cosido horizontal de imágenes dentro de cada fila
3) Cosido vertical de las filas ya unidas horizontalmente

Todo el proceso se apoya en OpenCV (ORB + matching + RANSAC) para inferir la
geometría relativa entre imágenes.

────────────────────────────────────────────────────────────
1. ESTIMACIÓN DE DESPLAZAMIENTOS (ORB + RANSAC)
────────────────────────────────────────────────────────────

La función estimate_shift() calcula el desplazamiento relativo (dx, dy) entre dos
imágenes consecutivas:

- Conversión a escala de grises
- Detección de puntos clave ORB
- Matching de descriptores con distancia Hamming
- Filtrado con el criterio de Lowe (ratio test)
- Estimación de una transformación afín parcial usando RANSAC
- Extracción del desplazamiento (dx, dy)
- Cálculo de una métrica de confianza basada en el porcentaje de inliers

Opcionalmente, el proceso puede visualizar los matches válidos para depuración.

────────────────────────────────────────────────────────────
2. COSIDO HORIZONTAL DE CADA FILA
────────────────────────────────────────────────────────────

Para cada fila:

- Se calculan los desplazamientos horizontales entre columnas consecutivas.
- Se aplican correcciones manuales específicas para casos problemáticos donde
  la estimación automática no es fiable.
- Se construye un lienzo incremental concatenando las partes no solapadas de
  cada imagen.
- El resultado de cada fila se guarda como una imagen intermedia:
      row_XX_stitched.png

Este paso produce una imagen por fila, ya unida horizontalmente.

────────────────────────────────────────────────────────────
3. COSIDO VERTICAL DE LAS FILAS
────────────────────────────────────────────────────────────

Una vez cosidas las filas:

- Se estiman desplazamientos verticales entre filas consecutivas.
- Se calcula una posición absoluta para cada fila en un sistema de coordenadas
  global, partiendo de una fila origen.
- Se descartan desplazamientos con baja confianza para evitar introducir
  geometría incorrecta.
- Se calcula el bounding box global que contiene todas las filas.
- Se crea un canvas final y se pintan las filas en sus posiciones absolutas.

El resultado final se guarda como:
    final_stitched.png

────────────────────────────────────────────────────────────
CONSIDERACIONES TÉCNICAS IMPORTANTES
────────────────────────────────────────────────────────────

- El sistema no asume alineación perfecta: toda la geometría se infiere.
- Se prioriza robustez frente a automatización total (ajustes manuales incluidos).
- El uso de ORB permite un buen equilibrio entre velocidad y calidad.
- El consumo de memoria puede ser elevado en las etapas finales.
- Existen dos implementaciones del cosido vertical:
  - stitch_rows_vertical_old(): versión incremental clásica
  - stitch_rows_vertical(): versión basada en posiciones absolutas globales

────────────────────────────────────────────────────────────
USO PREVISTO
────────────────────────────────────────────────────────────

Este script está pensado para:
- Reconstrucción técnica de grandes mosaicos
- Capturas automatizadas con solape
- Entornos controlados donde se acepta intervención manual puntual

No está diseñado como una solución genérica “plug & play”, sino como una
herramienta de reconstrucción geométrica controlada.
"""

import cv2
import numpy as np
import sys
import os


# ------------------------------------------------------------
# Estima el desplazamiento (dx, dy) entre dos imágenes
# usando ORB + matching + RANSAC
# Incluye visualización de los inliers
# ------------------------------------------------------------

def count_columns(row_dir):
    return len([
        f for f in os.listdir(row_dir)
        if f.startswith("col_") and f.endswith(".png")
    ])


def paint_inline_matchers_horizontal(matches, img1, kp1, img2, kp2, inliers):
    # --------------------------------------------------------
    # Visualización de matches válidos (inliers)
    # --------------------------------------------------------
    inlier_matches = [
        matches[i] for i in range(len(matches)) if inliers[i]
    ]
    vis = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Inlier matches (ORB + RANSAC)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def paint_inline_matchers_vertical(
        img1,
        img2,
        kp1,
        kp2,
        matches,
        inliers=None,
        max_width=1600,
        max_height=900
):
    """
    Muestra matches con img1 arriba e img2 abajo.
    Escala automáticamente para que quepa en pantalla.
    """

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas_w = max(w1, w2)
    canvas_h = h1 + h2

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    canvas[:h1, :w1] = img1
    canvas[h1:h1 + h2, :w2] = img2

    # Dibujar matches
    for i, m in enumerate(matches):
        if inliers is not None and not inliers[i]:
            continue

        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt

        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2 + h1))

        cv2.circle(canvas, pt1, 3, (0, 255, 0), -1)
        cv2.circle(canvas, pt2, 3, (0, 255, 0), -1)
        cv2.line(canvas, pt1, pt2, (255, 0, 0), 1)

    # Escalado automático
    scale = min(
        max_width / canvas_w,
        max_height / canvas_h,
        1.0
    )

    if scale < 1.0:
        canvas = cv2.resize(
            canvas,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    cv2.imshow("Vertical matches", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_manual_adjustements(row_x, col_y, dx_adj, conf_adj):
    # Ajustes manuales
    if row_x == 6:
        if col_y == 8:
            dx_adj = -1488
            conf_adj = 1
        elif col_y == 9:
            dx_adj = -1470
            conf_adj = 1
        elif col_y == 10:
            dx_adj = -1900
            conf_adj = 1
    if row_x == 7 and col_y == 9:
        dx_adj = -1690
        conf_adj = 1
    if row_x == 8:
        if col_y == 5:
            dx_adj = -1730
            conf_adj = 1
        elif col_y == 6:
            dx_adj = -1490
            conf_adj = 1
        if col_y == 14:
            dx_adj = -1518
            conf_adj = 1
    if row_x == 9:
        if col_y == 8:
            dx_adj = -1500
            conf_adj = 1
        elif col_y == 9:
            dx_adj = -1700
            conf_adj = 1
        if col_y == 14:
            dx_adj = -1537
            conf_adj = 1
        elif col_y == 15:
            dx_adj = -1689
            conf_adj = 1
    if row_x == 10 and col_y == 3:
        dx_adj = -1730
        conf_adj = 1
    if row_x == 11:
        if col_y == 6:
            dx_adj = -1679
            conf_adj = 1
        elif col_y == 11:
            dx_adj = -1713
            conf_adj = 1
    if row_x == 12:
        if col_y == 1:
            dx_adj = -1677
            conf_adj = 1
        elif col_y == 13:
            dx_adj = -1496
            conf_adj = 1
    if row_x == 13:
        if col_y == 3:
            dx_adj = -1910
            conf_adj = 1
        elif col_y == 7:
            dx_adj = -1684
            conf_adj = 1
    if row_x == 14:
        if col_y == 10:
            dx_adj = -1702
            conf_adj = 1
        elif col_y == 11:
            dx_adj = -1491
            conf_adj = 1
    if row_x == 15 and col_y == 7:
        dx_adj = -1521
        conf_adj = 1
    if row_x == 16:
        if col_y == 4:
            dx_adj = -1668
            conf_adj = 1
        elif col_y == 5:
            dx_adj = -1692
            conf_adj = 1
    if row_x == 17 and col_y == 9:
        dx_adj = -1670
        conf_adj = 1
    if row_x == 18 and col_y == 4:
        dx_adj = -1577
        conf_adj = 1
    if row_x == 19:
        if col_y == 6:
            dx_adj = -1682
            conf_adj = 1
        elif col_y == 9:
            dx_adj = -1665
            conf_adj = 1
        elif col_y == 17:
            dx_adj = -1669
            conf_adj = 1
    if row_x == 20 and col_y == 16:
        dx_adj = -1534
        conf_adj = 1
    if row_x == 21 and col_y == 9:
        dx_adj = -1530
        conf_adj = 1
    if row_x == 22:
        if col_y == 10:
            dx_adj = -1495
            conf_adj = 1
        elif col_y == 11:
            dx_adj = -1508
            conf_adj = 1
        elif col_y == 12:
            dx_adj = -1523
            conf_adj = 1
    if row_x == 24:
        if col_y == 8:
            dx_adj = -1487
            conf_adj = 1

    return dx_adj, conf_adj


def estimate_shift(img1, img2, min_matches=30, debug=False):
    """
    img1, img2: imágenes consecutivas (BGR o GRAY)
    devuelve:
      dx, dy, confianza
      y si debug=True:
      dx, dy, confianza, kp1, kp2, good_matches, inliers
    """

    # 1. Convertir a escala de grises
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1.copy()

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2.copy()

    # 2. Detector ORB
    orb = cv2.ORB_create(
        nfeatures=5000,
        scaleFactor=1.2,
        nlevels=8
    )

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        if debug:
            return 0, 0, 0.0, [], [], [], []
        return 0, 0, 0.0

    # 3. Matching (Hamming)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # 4. Lowe ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < min_matches:
        if debug:
            return 0, 0, 0.0, kp1, kp2, good, []
        return 0, 0, 0.0

    # 5. Extraer coordenadas
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # 6. Estimar transformación con RANSAC
    M, inliers = cv2.estimateAffinePartial2D(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )

    if M is None or inliers is None:
        if debug:
            return 0, 0, 0.0, kp1, kp2, good, []
        return 0, 0, 0.0

    # 7. Desplazamiento
    dx = float(M[0, 2])
    dy = float(M[1, 2])

    # 8. Confianza = porcentaje de inliers
    confidence = float(np.mean(inliers))

    if debug:
        paint_inline_matchers_vertical(img1, img2, kp1, kp2, good, inliers)
        return dx, dy, confidence, kp1, kp2, good, inliers

    return dx, dy, confidence


def stitch_row(row_x, col_y, col_y_max, dx_conf_list, input_dir, output_dir):
    canvas = None
    col_idx = col_y  # empezamos en col_y

    # row_dir = os.path.join(input_dir, f"row_{row_x:02d}")
    max_cols = col_y_max  # count_columns(row_dir)

    for dx, conf in dx_conf_list:

        if col_idx > max_cols:
            break

        if dx == 0 or conf == 0:
            print(f"col_idx {col_idx} without dx or confidence info")
            if canvas is not None or col_idx == max_cols:
                # Guardar resultado
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(
                    output_dir, f"row_{row_x:02d}_stitched.png"
                    # output_dir, f"row_{row_x:02d}_{col_idx}_stitched.png"
                )
                if canvas is not None:
                    cv2.imwrite(out_path, canvas)
            canvas = None
            col_idx += 1
            continue

        if col_idx > max_cols:
            break

        first_path = None
        if canvas is None:
            # Cargar primera imagen
            first_path = os.path.join(
                input_dir, f"row_{row_x:02d}", f"col_{col_idx:02d}.png"
            )
            canvas = cv2.imread(first_path, cv2.IMREAD_COLOR)
            h, _, c = canvas.shape

        if canvas is None:
            raise RuntimeError(f"No se pudo leer {first_path}")

        col_idx += 1
        if col_idx > max_cols:
            break

        img_path = os.path.join(
            input_dir, f"row_{row_x:02d}", f"col_{col_idx:02d}.png"
        )

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"WARNING")
            continue

        step = int(abs(dx))
        img_w = img.shape[1]

        print(f"row_x({row_x}),col_idx({col_idx}): step = {step}, img_w = {img_w}")

        overlap = img_w - step
        if overlap < 0:
            overlap = 0

        new_part = img[:, overlap:, :]

        # Ampliar lienzo y pegar
        canvas = np.concatenate((canvas, new_part), axis=1)

    if canvas is not None:
        # Guardar resultado
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"row_{row_x:02d}_stitched.png")
        cv2.imwrite(out_path, canvas)

    # return out_path


def stitch_rows_vertical_old(row_y, row_y_max, dx_conf_list, dy_conf_list, input_dir, output_dir):
    """
    Une filas ya cosidas horizontalmente usando desplazamiento 2D (dx, dy).

    Cada fila se coloca en su posición real en el lienzo.
    """

    assert len(dx_conf_list) == len(dy_conf_list)

    # ------------------------------------------------------------
    # 1. Cargar primera fila
    # ------------------------------------------------------------
    row_idx = row_y
    first_path = os.path.join(input_dir, f"row_{row_idx:02d}_stitched.png")
    first = cv2.imread(first_path)

    if first is None:
        raise RuntimeError(f"No se pudo leer {first_path}")

    h0, w0 = first.shape[:2]

    # ------------------------------------------------------------
    # 2. Posición inicial (origen arbitrario)
    # ------------------------------------------------------------
    cur_x = 0
    cur_y = 0

    # Guardamos posiciones absolutas de cada fila
    placements = [(cur_x, cur_y, first)]
    min_x, min_y = cur_x, cur_y
    max_x, max_y = cur_x + w0, cur_y + h0

    # ------------------------------------------------------------
    # 3. Calcular posiciones absolutas de todas las filas
    # ------------------------------------------------------------
    for i, ((dx, conf_x), (dy, conf_y)) in enumerate(
            zip(dx_conf_list, dy_conf_list)
    ):
        row_idx += 1
        if row_idx > row_y_max:
            break

        # filtro mínimo de confianza
        if conf_x < 0.2 or conf_y < 0.2:
            continue

        img_path = os.path.join(
            input_dir, f"row_{row_idx:02d}_stitched.png"
        )
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # avance real
        cur_x += int(round(dx))
        cur_y += abs(int(round(dy)))

        placements.append((cur_x, cur_y, img))

        min_x = min(min_x, cur_x)
        min_y = min(min_y, cur_y)
        max_x = max(max_x, cur_x + w)
        max_y = max(max_y, cur_y + h)

    # ------------------------------------------------------------
    # 4. Crear lienzo final (bounding box)
    # ------------------------------------------------------------
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # ------------------------------------------------------------
    # 5. Pegar filas en sus posiciones absolutas
    # ------------------------------------------------------------
    for x, y, img in placements:
        h, w = img.shape[:2]
        cx = x - min_x
        cy = y - min_y

        canvas[cy:cy + h, cx:cx + w] = img

    # ------------------------------------------------------------
    # 6. Guardar resultado
    # ------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "final_stitched.png")
    cv2.imwrite(out_path, canvas)


def stitch_rows_vertical(row_y, row_y_max, dx_conf_list, dy_conf_list, input_dir, output_dir, min_conf=0.2):
    # ---------------------------------------------------------
    # 1. Cargar imágenes de filas
    # ---------------------------------------------------------
    rows = {}
    widths = {}
    heights = {}

    for r in range(row_y, row_y_max + 1):
        path = os.path.join(input_dir, f"row_{r:02d}_stitched.png")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"No se pudo leer {path}")

        rows[r] = img
        h, w = img.shape[:2]
        widths[r] = w
        heights[r] = h

    # ---------------------------------------------------------
    # 2. Calcular posiciones absolutas
    # ---------------------------------------------------------
    # Sistema de coordenadas global
    # fila row_y es el origen (0,0)
    positions = {}
    positions[row_y] = (0.0, 0.0)

    list_idx = 0
    for r in range(row_y + 1, row_y_max + 1):
        prev_x, prev_y = positions[r - 1]

        dx, dx_conf = dx_conf_list[list_idx]
        dy, dy_conf = dy_conf_list[list_idx]

        # Si la confianza es mala, NO inventamos geometría
        if dx_conf < min_conf:
            dx = 0.0
        if dy_conf < min_conf:
            dy = 0.0

        positions[r] = (
            prev_x + dx,
            prev_y + dy
        )

        list_idx += 1

    # ---------------------------------------------------------
    # 3. Bounding box global
    # ---------------------------------------------------------
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for r, (x, y) in positions.items():
        w = widths[r]
        h = heights[r]

        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    # ---------------------------------------------------------
    # 4. Crear canvas final
    # ---------------------------------------------------------
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # ---------------------------------------------------------
    # 5. Pintar filas en coordenadas absolutas
    # ---------------------------------------------------------
    for r, img in rows.items():
        x, y = positions[r]

        cx = int(round(x - min_x))
        cy = int(round(y - min_y))

        h, w = img.shape[:2]

        canvas[cy:cy + h, cx:cx + w] = img

    # ---------------------------------------------------------
    # 6. Guardar resultado
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "final_stitched.png")
    cv2.imwrite(out_path, canvas)


def join_rows():
    row_start = 1
    row_max = 26  # 26
    col_start = 1
    col_max = 21  # 21

    for row in range(row_start, row_max + 1):
        row_dx_conf = []
        for col in range(col_start, col_max):
            img1 = cv2.imread(f"input\\row_{row:02d}\\col_{col:02d}.png", cv2.IMREAD_COLOR)
            img2 = cv2.imread(f"input\\row_{row:02d}\\col_{col + 1:02d}.png", cv2.IMREAD_COLOR)

            if img1 is None or img2 is None:
                print("Error leyendo imágenes")
                sys.exit(1)

            # dx, dy, conf, kp1, kp2, matches, inliers = estimate_shift(img1, img2, debug=True)
            dx, dy, conf = estimate_shift(img1, img2)

            dx, conf = get_manual_adjustements(row, col, dx, conf)

            ####################################################################

            row_dx_conf.append((dx, conf))

            print(f"row = {row}, col = {col}: dx = {dx:.2f} px, dy = {dy:.2f} px, confianza = {conf:.3f}")

            # paint_inline_matchers_horizontal(matches, img1, kp1, img2, kp2, inliers)

        print(f"----------------------")
        stitch_row(row_x=row, col_y=col_start, col_y_max=col_max, dx_conf_list=row_dx_conf, input_dir="input", output_dir="output_rows")


def join_rows_vertical():
    row_start = 3  # 3
    row_max = 24  # 24

    row_dx_conf = []
    row_dy_conf = []
    for row in range(row_start, row_max):

        img1 = cv2.imread(f"output_rows\\row_{row:02d}_stitched.png", cv2.IMREAD_COLOR)
        img2 = cv2.imread(f"output_rows\\row_{row + 1:02d}_stitched.png", cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            print("Error leyendo imágenes")
            sys.exit(1)

        # dx, dy, conf, kp1, kp2, matches, inliers = estimate_shift(img1, img2, debug=True)
        dx, dy, conf = estimate_shift(img1, img2)

        ####################################################################

        row_dx_conf.append((dx, conf))
        row_dy_conf.append((dy, conf))

        print(f"row = {row}: dx = {dx:.2f} px, dy = {dy:.2f} px, confianza = {conf:.3f}")

    print(f"----------------------")
    # stitch_rows_vertical(row_y=row_start, row_y_max=row_max, dx_conf_list=row_dy_conf, dy_conf_list=row_dy_conf, input_dir="output_rows",
    #                      output_dir="output_final")
    stitch_rows_vertical_old(row_y=row_start, row_y_max=row_max, dx_conf_list=row_dy_conf, dy_conf_list=row_dy_conf, input_dir="output_rows",
                             output_dir="output_final")


if __name__ == "__main__":
    join_rows()
    join_rows_vertical()
