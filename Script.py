# from hmac import trans_36
import cv2 as cv
import Transformations as tf

image1N = input("Ingrese la dirección de la primera imagen: ")
image2N = input("Ingrese la dirección de la segunda imagen: ")

img_1 = cv.imread(image1N)
img_2 = cv.imread(image2N)
key, a = tf.start(img_1, img_2)
cv.imshow(key, a)
cv.waitKey(0)
