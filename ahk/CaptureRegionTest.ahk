/*
============================================================
Script: capture_region_single.ahk
Propósito: Captura una región fija de la pantalla y guarda
           el resultado como PNG de alta calidad (100).
Detalles:
- Utiliza GDI+ a través de la librería Gdip_All.ahk
- Configuración inicial permite definir:
    - Esquina superior izquierda (x1, y1)
    - Tamaño de la región (w, h)
    - Desplazamiento opcional (dx, dy) y delay
- El script crea automáticamente la carpeta 'captures'
- Guardado de la imagen final en la ruta ./captures/ahk_test.png
- Incluye liberación de recursos GDI+ al finalizar
Uso:
- Ejecutar el script, espera un mensaje indicando que la imagen
  ha sido guardada.
============================================================
*/

#Include ./libs/Gdip_All.ahk
;https://github.com/marius-sucan/AHK-GDIp-Library-Compilation

#NoEnv
#SingleInstance Force
SetBatchLines, -1
CoordMode, Pixel, Screen
CoordMode, Mouse, Screen

; ===== CONFIGURACIÓN =====
dx := 300      ; píxeles scroll horizontal
dy := 200      ; píxeles scroll vertical
delay := 300   ; ms entre acciones
x1 := 400      ; esquina superior izquierda
y1 := 200
w := 600      ; ancho
h := 400      ; alto
x2 := x1 + w
y2 := y1 + h

outputDir := A_ScriptDir . "\captures\"
delay := 300

FileCreateDir, %outputDir%

pToken := Gdip_Startup()
if !pToken
    ExitApp

; Capturar región de pantalla
pBitmap := Gdip_BitmapFromScreen(x1 "|" y1 "|" x2 "|" y2)

; Ruta del archivo PNG en el mismo directorio del script
FileName := outputDir . "ahk_test.png"

; Guardar como PNG (calidad 100)
Gdip_SaveBitmapToFile(pBitmap, FileName, 100)

; Liberar recursos
Gdip_DisposeImage(pBitmap)

MsgBox, 64, Listo, Imagen guardada en: %FileName%


Gdip_Shutdown(pToken)

ExitApp


; ===== FUNCIÓN DE CAPTURA =====
CaptureRegion(x, y, w, h, file)
{
    hbm := CreateDIBSection(w, h)
    hdc := CreateCompatibleDC()
    obm := SelectObject(hdc, hbm)

    hdcSrc := GetDC(0)
    BitBlt(hdc, 0, 0, w, h, hdcSrc, x, y, 0x00CC0020)
    ReleaseDC(0, hdcSrc)

    MsgBox, %file%
    pBitmap := Gdip_CreateBitmapFromHBITMAP(hbm)
    Gdip_SaveBitmapToFile(pBitmap, "pollas1000.png")

    SelectObject(hdc, obm)
    DeleteObject(hbm)
    DeleteDC(hdc)
    Gdip_DisposeImage(pBitmap)
}
