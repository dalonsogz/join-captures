"""
Este script implementa una utilidad de diagnóstico para estimar el desplazamiento
relativo entre dos imágenes consecutivas utilizando visión artificial.

Está diseñado como una herramienta de prueba y depuración del algoritmo de
estimación de desplazamiento empleado posteriormente en procesos de cosido
(stitching) de imágenes más complejos.

────────────────────────────────────────────────────────────
OBJETIVO
────────────────────────────────────────────────────────────

- Calcular el desplazamiento horizontal (dx) y vertical (dy) entre dos imágenes.
- Estimar una métrica de confianza asociada al resultado.
- Permitir la visualización de los matches válidos (inliers) para análisis visual.

────────────────────────────────────────────────────────────
MÉTODO UTILIZADO
────────────────────────────────────────────────────────────

El cálculo del desplazamiento se basa en el siguiente pipeline:

1. Conversión de las imágenes de entrada a escala de grises.
2. Detección de puntos clave y descriptores mediante ORB.
3. Emparejamiento de descriptores usando distancia Hamming.
4. Filtrado de matches con el criterio de Lowe (ratio test).
5. Estimación de una transformación afín parcial mediante RANSAC.
6. Extracción del desplazamiento (dx, dy) a partir de la matriz estimada.
7. Cálculo de la confianza como el porcentaje de inliers.

────────────────────────────────────────────────────────────
MODO DEPURACIÓN
────────────────────────────────────────────────────────────

Cuando debug=True:

- Se devuelven, además de dx y dy:
  - keypoints de ambas imágenes
  - lista de matches filtrados
  - máscara de inliers resultante de RANSAC
- El script permite visualizar gráficamente los matches válidos para
  inspeccionar la calidad de la estimación.

────────────────────────────────────────────────────────────
USO PREVISTO
────────────────────────────────────────────────────────────

Este script se utiliza para:
- Validar el comportamiento del algoritmo de estimación de desplazamiento.
- Ajustar parámetros (número de features, ratio test, umbral RANSAC).
- Identificar fallos de correspondencia entre imágenes concretas.

No está pensado como herramienta final de producción, sino como apoyo técnico
para desarrollo y ajuste fino del pipeline de stitching.
"""

import cv2
import numpy as np
import sys


def estimate_shift(img1_p, img2_p, min_matches=30, debug=False):
    """
    img1, img2: imágenes consecutivas (BGR o GRAY)
    devuelve:
      dx, dy, confianza
      y si debug=True:
      dx, dy, confianza, kp1, kp2, good_matches, inliers
    """

    # 1. Convertir a escala de grises
    if len(img1_p.shape) == 3:
        gray1 = cv2.cvtColor(img1_p, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1_p.copy()

    if len(img2_p.shape) == 3:
        gray2 = cv2.cvtColor(img2_p, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2_p.copy()

    # 2. Detector ORB
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)

    kp1_f, des1 = orb.detectAndCompute(gray1, None)
    kp2_f, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        if debug:
            return 0, 0, 0.0, [], [], [], []
        return 0, 0, 0.0

    # 3. Matching (Hamming)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_f = bf.knnMatch(des1, des2, k=2)

    # 4. Lowe ratio test
    good = []
    for m, n in matches_f:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < min_matches:
        if debug:
            return 0, 0, 0.0, kp1_f, kp2_f, good, []
        return 0, 0, 0.0

    # 5. Extraer coordenadas
    pts1 = np.float32([kp1_f[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2_f[m.trainIdx].pt for m in good])

    # 6. Estimar transformación con RANSAC
    M, inliers_f = cv2.estimateAffinePartial2D(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )

    if M is None or inliers_f is None:
        if debug:
            return 0, 0, 0.0, kp1_f, kp2_f, good, []
        return 0, 0, 0.0

    # 7. Desplazamiento
    dx_f = float(M[0, 2])
    dy_f = float(M[1, 2])

    # 8. Confianza = porcentaje de inliers
    confidence = float(np.mean(inliers_f))

    if debug:
        return dx_f, dy_f, confidence, kp1_f, kp2_f, good, inliers_f

    return dx_f, dy_f, confidence


def paint_inline_matchers(matches_p, img1_p, kp1_p, img2_p, kp2_p):
    # --------------------------------------------------------
    # Visualización de matches válidos (inliers)
    # --------------------------------------------------------
    inlier_matches = [
        matches_p[i] for i in range(len(matches_p)) if inliers[i]
    ]
    vis = cv2.drawMatches(
        img1_p, kp1_p,
        img2_p, kp2_p,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Inlier matches (ORB + RANSAC)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    sys.argv = []
    sys.argv.append("estimate_shift")
    sys.argv.append("input\\row_10\\col_04.png")
    sys.argv.append("input\\row_10\\col_05.png")

    if len(sys.argv) != 3:
        print("Uso: python estimate_shift_debug.py img1.png img2.png")
        sys.exit(1)

    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        print("Error leyendo imágenes")
        sys.exit(1)

    dx, dy, conf, kp1, kp2, matches, inliers = estimate_shift(img1, img2, debug=True)

    print(f"dx = {dx:.2f} px")
    print(f"dy = {dy:.2f} px")
    print(f"confianza = {conf:.3f}")

    # --------------------------------------------------------
    # Visualización de matches válidos (inliers)
    # --------------------------------------------------------

    paint_inline_matchers(matches, img1, kp1, img2, kp2)
