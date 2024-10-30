import cv2
import numpy as np

def gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def clahe(img):
    if len(img.shape) == 3:  # Verifica se a imagem é colorida
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return img

def sobel(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, 7, 35, 35)
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobel = cv2.bitwise_or(sobelX, sobelY)
    return sobel

def canny(img):
    # Canny
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 30, 150)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def bilateral(img):
    #img = img[::2,::2] # Diminui a imagem
    img = cv2.bilateralFilter(img, 11, 77, 77)
    return img

def normal(img):
    return img