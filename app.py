import cv2
import numpy as np

original = cv2.imread("dibujo.jpeg")
cv2.imshow("Original", original)

gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)


gauss = cv2.GaussianBlur(gris, (5,5), 0)

# Detectar bordes con Canny
canny = cv2.Canny(gauss, 50, 150)
cv2.imshow("Bordes Canny", canny)

# Detectar círculos (nodos del grafo)
circulos = cv2.HoughCircles(gauss, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=50, param2=30, minRadius=5, maxRadius=50)


if circulos is not None:
    circulos = np.uint16(np.around(circulos))
    for i in circulos[0, :]:
        # Dibujar el círculo
        cv2.circle(original, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Dibujar el centro
        cv2.circle(original, (i[0], i[1]), 2, (0, 0, 255), 3)

    print(f"{len(circulos[0])} nodos en el grafo.")

# Detectar líneas (posibles aristas del grafo)
lineas = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)

# Dibujar las líneas detectadas
if lineas is not None:
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        cv2.line(original, (x1, y1), (x2, y2), (255, 0, 0), 2)

    print(f"{len(lineas)} aristas en el grafo.")

# Mostrar la imagen con nodos y aristas detectados
cv2.imshow("Grafo Detectado", original)

cv2.waitKey(0)
cv2.destroyAllWindows()
