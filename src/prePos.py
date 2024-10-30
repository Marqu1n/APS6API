import cv2
import numpy as np

def resize(image,largura_nova,altura_nova):
    original_height, original_width = image.shape[:2]
    scale = min(largura_nova / original_width, altura_nova / original_height)
    l = int(original_width * scale)
    a = int(original_height * scale)
    tamanho_novo = (l, a)
    img_redimensionada = cv2.resize(image, tamanho_novo,  interpolation = cv2.INTER_AREA)
    result_image = np.zeros((altura_nova, largura_nova, 3), dtype=np.uint8)
    x_offset = (largura_nova - l) // 2
    y_offset = (altura_nova - a) // 2
    result_image[y_offset:y_offset + a, x_offset:x_offset + l] = img_redimensionada
    
    return result_image

def gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def clahe(img):
    if len(img.shape) == 3:  # Verifica se a imagem é colorida
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return img
    
def laplaciano(img):
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    img = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return lap

def sobel(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobel = cv2.bitwise_or(sobelX, sobelY)
    sobel = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    return sobel

def canny(img):
    # Canny
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 30, 150)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def bilateral(img):
    #img = img[::2,::2] # Diminui a imagem
    img = cv2.bilateralFilter(img, 7, 35, 35)
    return img

def normal(img):
    return img