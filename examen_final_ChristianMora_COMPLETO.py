# Christian Rafael Mora Parga, parcial final Procesamiento Imágenes y vídeo 2021. 24/Nov/2021
import cv2
import os
import sys
import numpy as np

if __name__ == '__main__':
    # ==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.
    # PUNTO 1, porcentaje césped
    # ==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.

    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)

    # Se calcula el histograma de Hue
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])

    # Máximo y mínimo del historial calculado
    max_val, max_pos = hist_hue.max(), int(hist_hue.argmax())

    # Se genera máscara a partir de los límites (inf y sup) calculados
    lim_inf, lim_sup = (max_pos - 10, 0, 0), (max_pos + 10, 255, 255)
    mask = cv2.inRange(image_hsv, lim_inf, lim_sup)
    # Se hacen operaciones morfológicas para segmentar mejor la sección de césped y lo demás
    W = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * W + 1, 2 * W + 1))
    mask_eroded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    W = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * W + 1, 2 * W + 1))
    mask_dilated = cv2.morphologyEx(mask_eroded, cv2.MORPH_CLOSE, kernel)

    # Se halla la cantidad de elementos true y false, que corresponden respectivamente a la zona de
    # interés (césped) blanco y otros en negro
    unique, counts = np.unique(mask_dilated, return_counts=True)
    num_rows, num_cols = (mask_dilated.shape[0], mask_dilated.shape[1])
    #Total de pixeles (100 %)
    TP = num_rows*num_cols #o hacer counts[0]+counts[1]

    #Número de pixeles en el par de zonas, césped y otros:
    pix_otros, pix_cesped = counts[0], counts[1]
    porcen_otros_pix, porcen_cesped_pix = 100 * (pix_otros / TP), 100 * (pix_cesped/TP)
    print('Porcentaje aproximado de pixeles de césped es: ', round(porcen_cesped_pix, 1), '% y de otros es: ',
          round(porcen_otros_pix, 1), '%')

    cv2.imshow("Punto 1, imagen binaria", mask_dilated)
    cv2.waitKey(0)

    # ==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.
    # PUNTO 2, reconocimiento de jugadores/arbitro
    # ==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2))
    mask_OPEN = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    #Uso del algoritmo de Canny, para hallar bordes:
    s = 0.033
    # calcular la mediana de las intensidades de píxeles de un solo canal
    v = np.median(mask_OPEN)
    # aplicar la detección automática de bordes Canny utilizando la mediana calculada
    inferior, superior = int(max(0, (1.0 - s) * v)), int(min(255, (1.0 + s) * v))

    img_bordes = cv2.Canny(mask_OPEN, inferior, superior)

    (contornos, _) = cv2.findContours(mask_OPEN, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_personas = 0
    copia_image = image.copy()
    for contorno in contornos:
        area_jugador = cv2.contourArea(contorno) # se hallan las áreas de cada contorno hallado en la imagen

        # Regla de decisión, para mostrar solo las áreas de interés
        if (area_jugador > 700) and (area_jugador < 4000):
            num_personas += 1

            x, y, w, h = cv2.boundingRect(contorno)
            cv2.rectangle(copia_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    print('El número de jugadores/arbitro hallados en la imagen es: ', num_personas)
    cv2.imshow("Punto 2", copia_image)
    cv2.waitKey(0)

    # ==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.
    # PUNTO 3, recta paralela
    # ==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.

    copia2_image = image.copy()
    print('Seleccione 3 puntos, los dos primeros para definir una recta, y el tercero para generar una recta paralela '
          'a la primera, oprima "x" apenas haya seleccionado los 3 puntos')

    points = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    points_a = []

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click)

    point_counter = 0
    while True:
        cv2.imshow("Image", copia2_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            points_a = points.copy()
            points = []
            break
        if len(points) > point_counter:
            point_counter = len(points)
            cv2.circle(copia2_image, (points[-1][0], points[-1][1]), 3, [255, 0, 0], -1)

    N = len(points_a)
    assert N == 3, 'Se requieren  3 puntos por imagen (2  de la primera línea, 1  nueva paralela)'

    cv2.line(copia2_image, (points_a[0]), (points_a[1]), (255, 0, 0), thickness=3, lineType=8)
    m = (points_a[0][1] - points_a[1][1]) / (points_a[0][0] - points_a[1][0])

    b1 = round(points_a[0][1] - m*points_a[0][0]) # o b2 = round(points_a[1][1] - m1 * points_a[1][0])
    xf = copia2_image.shape[1]
    yf = round(m*xf + b1)

    c1, c2 = -1, -1
    homogenea_recta1 = (m / c1, b1 / c1, c1)

    cv2.line(copia2_image, (0, b1), (xf, yf), (0, 0, 255), thickness=3, lineType=8)

    b2 = round(points_a[2][1] - m*points_a[2][0])
    yf2 = round(m * xf + b2)
    homogenea_recta2 = (m / c2, b2 / c1, c2)

    cv2.line(copia2_image, (0, b2), (xf, yf2), (0, 255, 255), thickness=3, lineType=8)

    count = 1
    for punto in points_a:
        cv2.circle(copia2_image, punto, radius=7, color=(255, 255, 0), thickness=-1)
        cv2.putText(copia2_image, f"P{count}", punto, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        count += 1
    print('coodenadas homogenas de ambas rectas: y1 = ', homogenea_recta1, ', y2 = ', homogenea_recta2)
    cv2.imshow("Punto 3", copia2_image)
    cv2.waitKey(0)