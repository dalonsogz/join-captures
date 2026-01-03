/*
============================================================
Script: capture_full_mosaic.ahk
Propósito: Captura automáticamente un mosaico completo de
           múltiples filas y columnas de la pantalla.
Detalles:
- Utiliza GDI+ y MouseClickDrag para recorrer la región a capturar
- Configuración inicial permite definir:
    - Coordenadas de la región de captura (x1, y1, x2, y2)
    - Número de filas y columnas (nRows, nCols)
    - Delay entre acciones
- Cada captura se guarda como PNG con nombre:
    row_col_ahk_test.png en la carpeta ./captures/
- El script realiza scroll horizontal y vertical para recorrer todo el mosaico
- Incluye funciones auxiliares:
    - MouseClickDragCustom: movimiento y clic del ratón
    - DrawCaptureArea y DrawRectangle/Circle: visualización de área
- Manejo seguro de GDI+ y liberación de recursos al finalizar
- Esc + salir cierra el script inmediatamente
Uso:
- Ejecutar el script; espera unos segundos y empieza la captura
============================================================
*/

#Include ./libs/Gdip_All.ahk
;https://github.com/marius-sucan/AHK-GDIp-Library-Compilation

#SingleInstance Force
#NoEnv
SetBatchLines -1

; CONFIGURACIÓN
x1 := 285
y1 := 230
x2 := 2230
y2 := 1150
w := x2-x1      ; ancho
h := y2-y1      ; alto

; POSICIONES BASE
startRow := 1
nRows := 23     ; número de filas (23)
nCols := 17     ; capturas por fila (17)
delay := 500    ; ms entre acciones

outputDir := A_ScriptDir . "\captures\"
FileCreateDir, %outputDir%

If !pToken := Gdip_Startup()
{
	MsgBox "Gdiplus failed to start. Please ensure you have gdiplus on your system"
	ExitApp
}
OnExit("ExitFunc")

;DrawCaptureArea(2560,1440,x1,y1,x2,y2,w,h)
Sleep, 3000

Loop, %nRows%
{

    if (startRow<=A_Index)
    {

        row := A_Index
        Loop, %nCols%
        {
            ; Capturar región de pantalla
            ;MsgBox, 64, Info, %x1% "|" %y1% "|" %x2% "|" %y2%
            pBitmap := Gdip_BitmapFromScreen(x1 "|" y1 "|" w "|" h)

            ; Ruta del archivo PNG en el mismo directorio del script
            FileName := outputDir . row . "_" . A_Index . "_ahk_test.png"

            ; Guardar como PNG (calidad 100)
            Gdip_SaveBitmapToFile(pBitmap, FileName, 100)

            ; Liberar recursos
            Gdip_DisposeImage(pBitmap)

            SoundBeep, 7500, 50

            ; Scroll horizontal
            MouseClickDragCustom(x2,y1,x1,y1,delay)
        }

        Loop, %nCols%
        {
            MouseClickDragCustom(x1,y1,x2,y1,0)
        }
        Sleep, 1000
    }

    ; Scroll vertical al siguiente nivel
    MouseClickDragCustom(x2-200,y2,x2-200,y1,delay)
    Sleep, %delay%
}

Gdip_Shutdown(pToken)
Return

MouseClickDragCustom(x_ini,y_ini,x_end,y_end,delay_up)
{
    ; MsgBox, 64, MouseClickDragCustom, %x_ini%,%y_ini% . %x_end%,%y_end%
    MouseMove, x_ini, y_ini, 0
    Send, {LButton down}
    MouseMove, x_end,y_end, 5
    Sleep, delay_up
    Send, {LButton up}
}

DrawCaptureArea(width,height,x1,y1,x2,y2,w,h)
{
    drwArea:=SetDrawingArea(width,height)
    DrawCircle(drwArea,x1,y1,10)
    DrawCircle(drwArea,x2,y2,10)
    DrawRectangle(drwArea,x1,y1,w,h)
    UpdateDrawingArea(drwArea)
}

SetDrawingArea(width,height)
{
    ; Create a layered window (+E0x80000 : must be used for UpdateLayeredWindow to work!) that is always on top (+AlwaysOnTop), has no taskbar entry or caption
    Gui, 1: -Caption +E0x80000 +LastFound +AlwaysOnTop +ToolWindow +OwnDialogs
    Gui, 1: Show, NA

    ; Get a handle to this window we have created in order to update it later
    hwnd1 := WinExist()

    ; Create a gdi bitmap with width and height of what we are going to draw into it. This is the entire drawing area for everything
    hbm := CreateDIBSection(width, height)

    ; Get a device context compatible with the screen
    hdc := CreateCompatibleDC()

    ; Select the bitmap into the device context
    obm := SelectObject(hdc, hbm)

    ; Get a pointer to the graphics of the bitmap, for use with drawing functions
    G := Gdip_GraphicsFromHDC(hdc)

    drwArea := [G,hwnd1,hdc,hbm,obm,width,height]

    return drwArea
}

UpdateDrawingArea(drwArea)
{
    G := drwArea[1]
    hwnd1:= drwArea[2]
    hdc:= drwArea[3]
    hbm:= drwArea[4]
    obm:= drwArea[5]
    Width:= drwArea[6]
    Height:= drwArea[7]

    ;MsgBox, 64, Listo, test

    ; Update the specified window we have created (hwnd1) with a handle to our bitmap (hdc), specifying the x,y,w,h we want it positioned on our screen
    ; So this will position our gui at (0,0) with the Width and Height specified earlier
    UpdateLayeredWindow(hwnd1, hdc, 0, 0, Width, Height)

    ; Select the object back into the hdc
    SelectObject(hdc, obm)

    ; Now the bitmap may be deleted
    DeleteObject(hbm)

    ; Also the device context related to the bitmap may be deleted
    DeleteDC(hdc)

    ; The graphics may now be deleted
    Gdip_DeleteGraphics(G)
}

DrawCircle(drwArea,x,y,r)
{
    G := drwArea[1]

    ; Create a fully opaque red brush (ARGB = Transparency, red, green, blue) to draw a circle
    pBrush := Gdip_BrushCreateSolid(0xffff0000)

    ; Fill the graphics of the bitmap with an ellipse using the brush created
    Gdip_FillEllipse(G, pBrush, x, y, r, r)

    ; Delete the brush as it is no longer needed and wastes memory
    Gdip_DeleteBrush(pBrush)
}

DrawRectangle(drwArea,x1,y1,width,height)
{
    G := drwArea[1]

    ; Create a slightly transparent (66) blue brush (ARGB = Transparency, red, green, blue) to draw a rectangle
    pBrush := Gdip_BrushCreateSolid(0x330000aa)

    ; Fill the graphics of the bitmap with a rectangle using the brush created
    Gdip_FillRectangle(G, pBrush, x1, y1, width, height)

    ; Delete the brush as it is no longer needed and wastes memory
    Gdip_DeleteBrush(pBrush)
}

;#######################################################################

ExitFunc(ExitReason, ExitCode) {
   global
   ; gdi+ may now be shutdown on exiting the program
   Gdip_Shutdown(pToken)
}

~Esc::
   ExitApp
Return