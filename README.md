# Image Mosaic Stitching Toolkit (ORB + RANSAC)

Este proyecto implementa un conjunto de scripts en Python para la **reconstrucción de grandes mosaicos de imágenes** a partir de capturas parciales con solape, utilizando técnicas de visión artificial basadas en **OpenCV (ORB + matching + RANSAC)**.

El objetivo principal es **recomponer imágenes de gran tamaño** a partir de múltiples capturas organizadas en filas y columnas, donde la alineación no es perfecta y debe inferirse automáticamente.

---

## Visión general del proyecto

El flujo completo se divide en **cuatro scripts**, cada uno con una responsabilidad clara:

1. Estimación de desplazamientos entre imágenes  
2. Depuración visual de correspondencias  
3. Cosido horizontal y vertical basado en desplazamientos  
4. Ensamblado final de mosaicos gigantes  

El proyecto prioriza:
- Corrección geométrica
- Robustez frente a errores locales
- Control manual en casos problemáticos
- Claridad técnica frente a soluciones “caja negra”

---

## Scripts incluidos

### 1. `estimate_shift.py`  
**(estimación de desplazamiento y depuración visual)**

Herramienta de diagnóstico para estimar el desplazamiento relativo entre dos imágenes consecutivas.

**Funcionalidad:**
- Cálculo de desplazamiento horizontal (dx) y vertical (dy)
- Estimación de una métrica de confianza basada en inliers
- Visualización de matches válidos (inliers)
- Ajuste fino de parámetros ORB y RANSAC

**Uso previsto:**
- Depuración de casos problemáticos
- Validación visual de correspondencias
- Verificación de la calidad de la estimación automática

---

### 2. `joinCapturesRANSAC.py`  
**(pipeline completo de cosido automático)**

Script principal que implementa el **cosido horizontal por filas** y el **cosido vertical entre filas**, utilizando desplazamientos estimados automáticamente mediante visión artificial.

**Fases principales:**
- Estimación de desplazamientos entre columnas consecutivas
- Aplicación opcional de ajustes manuales
- Cosido horizontal incremental de cada fila
- Estimación de desplazamientos verticales entre filas
- Posicionamiento absoluto de cada fila en un sistema de coordenadas global
- Generación de la imagen final cosida

Incluye dos estrategias de cosido vertical:
- Enfoque incremental clásico
- Enfoque basado en posiciones absolutas y cálculo de bounding box global

---

### 3. `joinCaptures.py`  
**(ensamblado final de mosaicos gigantes con Pillow)**

Script orientado al **ensamblado determinista** de imágenes ya alineadas en un mosaico fijo de filas y columnas.

**Características:**
- Ensamblado fila a fila con control explícito de memoria
- Validación estricta del tamaño de cada imagen
- Exportación a PNG y JPG
- Configuración explícita de calidad y subsampling JPEG

Pensado para imágenes finales de **decenas de miles de píxeles por lado**.

---

## Requisitos

- Python 3.9 o superior
- OpenCV (`opencv-python`)
- NumPy
- Pillow

Instalación recomendada:

```bash
pip install opencv-python numpy pillow
```

## Consideraciones técnicas importantes

- Las imágenes finales pueden ocupar **varios gigabytes en memoria RAM**.
- Se recomienda un sistema con **16–32 GB de RAM** para trabajar con comodidad.
- JPEG no permite escritura incremental: la imagen completa debe residir en memoria.
- El sistema no asume alineación perfecta; la geometría se infiere automáticamente.
- Existen ajustes manuales explícitos para casos donde la estimación automática falla.

Este proyecto **no pretende ser una solución genérica de stitching**, sino una herramienta técnica controlada.

## Estructura de directorios típica

```text
project/
├── input/
│   ├── row_01/
│   │   ├── col_01.png
│   │   ├── col_02.png
│   │   └── ...
│   └── ...
├── output_rows/
│   └── row_XX_stitched.png
├── output_final/
│   └── final_stitched.png
├── scripts/
│   ├── estimate_shift_debug.py
│   ├── stitching_pipeline.py
│   ├── mosaic_builder_pillow.py
│   └── mosaic_builder_pillow_simple.py
└── README.md
```

## Limitaciones conocidas

- No se realiza corrección de perspectiva compleja (solo transformación afín parcial).
- No hay blending avanzado en zonas de solape.
- Los ajustes manuales están codificados explícitamente.
- No es adecuado para entornos con recursos limitados.

## Scripts AutoHotkey (AHK) incluidos

Este proyecto también incluye dos scripts AHK para la **captura automatizada de la pantalla**, que alimentan el pipeline de cosido de imágenes:

### 1. `capture_region_single.ahk`
- Captura una **región fija** de la pantalla y la guarda como PNG.
- Configurable: posición, tamaño, retraso entre acciones.
- Ideal para pruebas rápidas o captura de una sola sección.

### 2. `capture_full_mosaic.ahk`
- Captura un **mosaico completo** con varias filas y columnas.
- Utiliza GDI+ para la captura y MouseClickDrag para desplazamiento.
- Guarda todas las capturas con nombre estructurado: `row_col_ahk_test.png`.
- Permite la integración directa con los scripts de cosido en Python.

**Notas:**
- Ambos scripts requieren la librería GDI+ `Gdip_All.ahk`.
- Asegurarse de que la carpeta `captures` exista o se creará automáticamente.
- Los scripts incluyen liberación de recursos GDI+ al finalizar.


## Licencia

Este proyecto se distribuye bajo la licencia **MIT**.

Se permite el uso, modificación y redistribución del software, siempre que se conserve
el aviso de copyright y la licencia original.

Consulta el fichero `LICENSE` para más detalles.

## Aviso legal

Este software se proporciona **“tal cual”**, sin garantías de ningún tipo, explícitas
ni implícitas. El autor no se hace responsable de daños derivados del uso del software.

## Autor

Proyecto desarrollado con fines técnicos y experimentales.

Si reutilizas este código en proyectos derivados, se agradece la referencia.


