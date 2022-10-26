#!/usr/bin/env python
# coding: utf-8

import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# ## Lista de funciones:
# - resize(image) #Redimensionamiento
# - circle(image) #Círculo
# - rectangle(image) #Rectángulo
# - line(image) #Línea
# - text(image) #Texto
# - color_scale(image) #Escaña
# - desplazamiento(image) #Desplazar
# - rotate(image) #Rotar
# - flip(image) #Voltear
# - crop(image) #Cortar
# - threshold(image) #Filtrar
# - masking(image) #Enmascarar
# - brightness(image) #Brillo
# - contrast(image) #Contraste

def start(image1, image2):
    data = dict(locals())
    randomImage = random.choice(list(data.values()))
    my_list = [resize, circle, rectangle, line, text,
               color_scale, desplazamiento, rotate,
               flip, crop, threshold, masking,
               brightness, contrast]
    return random.choice(my_list)(randomImage)


def resize(image):
    imgOut = cv.resize(image, tuple(int(item / 2) for item in image.shape[:2][::-1]), interpolation=cv.INTER_AREA)
    cv.imwrite("resize.png", imgOut[:, :, ::-1])
    return "resize.png", imgOut


def circle(image):
    imgOut = image.copy()
    imgShape = image.shape[:2]
    cv.circle(imgOut, (int(imgShape[1] / 2), int(imgShape[0] / 2)), 100, (0, 0, 255), thickness=4, lineType=cv.LINE_AA)
    plt.imshow(imgOut[:, :, ::-1])
    cv.imwrite("circle.png", imgOut[:, :, ::-1])
    return "circle.png", imgOut


def rectangle(image):
    imgOut = image.copy()
    imgShape = image.shape[:2]
    cv.rectangle(imgOut, (int(imgShape[1] / 4), int(imgShape[0] / 4)),
                 (int(imgShape[1] / 2) + (int(imgShape[1] / 4)),
                  int(imgShape[0] / 2) + int(imgShape[0] / 4)), (0, 0, 255), 4)
    plt.imshow(imgOut[:, :, ::-1])
    cv.imwrite("rectangle.png", imgOut[:, :, ::-1])
    return "rectangle.png", imgOut


def line(image):
    imgOut = image.copy()
    imgShape = image.shape[:2]
    cv.line(imgOut, (int(imgShape[1] / 4), int(imgShape[0] / 4)),
            (int(imgShape[1] / 2) + (int(imgShape[1] / 4)),
             int(imgShape[0] / 2) + int(imgShape[0] / 4)), (0, 0, 255), thickness=4, lineType=cv.LINE_AA)
    plt.imshow(imgOut[:, :, ::-1])
    cv.imwrite("line.png", imgOut[:, :, ::-1])
    return "line.png", imgOut


def text(image):
    imgOut = image.copy()
    imgShape = image.shape[:2]
    text = "TEXTO"
    cv.putText(imgOut, text, (int(imgShape[1] / 2), int(imgShape[0] / 2)), cv.FONT_ITALIC, 1, (0, 0, 255), 4,
               cv.LINE_AA)
    plt.imshow(imgOut[:, :, ::-1])
    cv.imwrite("text.png", imgOut[:, :, ::-1])
    return "text.png", imgOut


def color_scale(image):
    plt.subplot(331)
    plt.imshow(image[:, :, ::-1], cmap='gray')
    plt.title("Original")
    plt.subplot(332)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cmap='gray')
    plt.title("Escala grises")
    plt.subplot(333)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2YCrCb), cmap='gray')
    plt.title("Escala YCrCb")
    plt.subplot(334)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2HSV), cmap='gray')
    plt.title("Escala HSV")
    plt.subplot(335)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2XYZ), cmap='gray')
    plt.title("Escala XYZ")
    plt.subplot(336)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2YUV), cmap='gray')
    plt.title("Escala YUV")
    plt.subplot(337)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2HLS), cmap='gray')
    plt.title("Escala HLS")
    plt.subplot(338)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2Luv), cmap='gray')
    plt.title("Escala Luv")
    plt.subplot(339)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2Lab), cmap='gray')
    plt.title("Escala Lab")
    plt.tight_layout()
    plt.show()
    return "escala.png", image


def desplazamiento(image):
    imgShape = image.shape[:2]
    M = np.float32([[1, 0, random.randint(0, imgShape[1])], [0, 1, random.randint(0, imgShape[0])]])
    imgOut = cv.warpAffine(image, M, image.shape[:2][::-1])
    cv.imwrite("desplazamiento.png", imgOut[:, :, ::-1])
    return "desplazamiento.png", imgOut


def rotate(image):
    imgShape = image.shape[:2]
    angle = random.randint(15, 180)
    M = cv.getRotationMatrix2D((int(imgShape[1] / 2), int(imgShape[0] / 2)), angle, 1)
    imgOut = cv.warpAffine(image, M, imgShape)
    cv.imwrite("rotate.png", imgOut[:, :, ::-1])
    return "rotate.png", imgOut


def flip(image):
    imgOut = cv.flip(image, random.randint(-1, 1))
    cv.imwrite("flip.png", imgOut[:, :, ::-1])
    return "flip.png", imgOut


def crop(image):
    imgShape = image.shape[:2]
    x = int(imgShape[1] / 4)
    y = int(imgShape[0] / 4)
    a = int(imgShape[1] / 2) + x
    b = int(imgShape[0] / 2) + y
    imgOut = image[y:b, x:a]
    cv.imwrite("recorte.png", imgOut[:, :, ::-1])
    return "recorte.png", imgOut


def threshold(image):
    image = cv.medianBlur(image, 5)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    (T, imgOut) = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imwrite("threshold.png", imgOut[:, ::-1])
    return "threshold.png", imgOut


def masking(image):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv.circle(mask, (145, 200), 100, 255, -1)
    imgOut = cv.bitwise_and(image, image, mask=mask)
    cv.imwrite("masking.png", imgOut[:, :, ::-1])
    return "masking.png", imgOut


def brightness(image):
    n = 200
    matrix = np.ones(image.shape, dtype='uint8') * n
    n = (0, 1)
    if n == 1:
        imgOut = cv.add(image, matrix)
    else:
        imgOut = cv.subtract(image, matrix)
    cv.imwrite("brightness.png", imgOut[:, :, ::-1])
    return "brightness.png", imgOut


def contrast(image):
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv.merge((cl, a, b))
    imgOut = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    cv.imwrite("contrast.png", imgOut[:, :, ::-1])
    return "contrast.png", imgOut
