"""Auxiliares compartilhados para contornos e operações com máscaras binárias."""
import cv2
import numpy as np


# Cria elemento estruturante elíptico, utilizado nas operações morfológicas.
# O tamanho é no mínimo 3 e sempre ímpar: int(size) | 1 ativa o último bit.
def kernel(size):
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, int(size) | 1),) * 2)


# Encontra contornos externos da máscara binária;
# CHAIN_APPROX_SIMPLE reduz pontos redundantes nas bordas.
def contours(mask):
    return cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)[0]


# Reconstrói regiões sólidas preenchendo seus contornos externos.
def filled(mask):
    result = np.zeros_like(mask)
    cv2.drawContours(result, contours(mask), -1, 255, cv2.FILLED)
    return result


# Cria uma máscara do tamanho desejado contendo somente um contorno preenchido.
def contour_mask(shape, contour):
    result = np.zeros(shape, np.uint8)
    cv2.drawContours(result, [contour], -1, 255, -1)
    return result
