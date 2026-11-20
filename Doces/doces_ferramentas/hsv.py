"""Segmentação por cor no espaço HSV.

H representa o matiz; S, a saturação; V, o brilho. As faixas selecionam
pixels de cada cor e geram máscaras binárias. No OpenCV, H usa 0..179;
S e V usam 0..255. Os limites principais são lidos do config.json.
"""
from __future__ import annotations

import cv2
import numpy as np

from .mask_utils import contours, kernel


# Segmentação por cor: devolve máscara branca (255) para pixels
# dentro do intervalo HSV e preta (0) para os demais.
def in_range(hsv, limits):
    return cv2.inRange(hsv, np.array(limits[0], np.uint8), np.array(limits[1], np.uint8))


def prepare_hsv(image):
    """Suaviza a imagem, converte BGR para HSV e compensa fotos escuras."""
    # Suaviza pequenos ruídos antes de distinguir as cores.
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    # Converte de BGR (OpenCV) para HSV: H=matiz, S=saturação, V=brilho.
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    # Realça contraste de brilho local para procurar a haste metálica.
    metal_highlight = value - cv2.GaussianBlur(value, (0, 0), 9)
    # Ajusta só o brilho V em cenas escuras; limita o ganho a 2,5
    # para não exagerar os ruídos e reflexos.
    gain = min(2.5, max(1.0, 220.0 / max(1, np.percentile(value, 95))))
    hsv[:, :, 2] = np.clip(value * gain, 0, 255).astype(np.uint8)

    return hsv, metal_highlight, gain


def pink_mask(hsv, config):
    """Seleciona o rosa e fecha pequenas falhas na máscara."""
    w = hsv.shape[1]
    pink = in_range(hsv, config["pink_hsv"])
    # O matiz HSV é circular: certos tons rosados podem aparecer perto de 0.
    # Só aproveita esses pixels perto de regiões já reconhecidas como rosa.
    wrapped = cv2.inRange(hsv, np.array([0, 40, 100]), np.array([5, 210, 255]))
    pink |= wrapped & cv2.dilate(pink, kernel(w / 50))
    # Fechamento = dilatação seguida de erosão; fecha pequenas falhas.
    pink = cv2.morphologyEx(pink, cv2.MORPH_CLOSE, kernel(w / 100))
    return pink


def tool_mask(hsv, config, gain):
    """Seleciona o amarelo do cabo e une suas manchas próximas."""
    w = hsv.shape[1]
    limits = [list(row) for row in config["tool_hsv"]]
    # Ajusta o brilho mínimo do amarelo à correção de iluminação.
    limits[0][2] = round(limits[0][2] / gain)
    # Une pedaços amarelos próximos, formando o corpo do cabo.
    yellow = cv2.morphologyEx(in_range(hsv, limits), cv2.MORPH_CLOSE, kernel(w / 43))
    return yellow


def gummy_mask(hsv, config):
    """Une as faixas de cores que podem pertencer às minhocas."""
    h, w = hsv.shape[:2]
    gummy = np.zeros((h, w), np.uint8)
    # Une cores quentes, vermelho, verde e azul em uma máscara.
    for key in ("warm_hsv", "red_wrap_hsv", "green_hsv", "blue_hsv"):
        gummy |= in_range(hsv, config[key])
    return gummy


def metal_mask(hsv, metal_highlight, corridor, contrast, saturation):
    """Seleciona pixels claros ou sombreados que podem formar a haste."""
    bright = (hsv[:, :, 1] < 90) & (metal_highlight > 18)
    shaded = (hsv[:, :, 1] < saturation) & (metal_highlight < -contrast)
    # A haste pode estar mais clara OU mais escura que sua vizinhança.
    metal = ((bright | shaded) & (hsv[:, :, 2] > 60) & corridor).astype(np.uint8) * 255
    return metal


def gummy_color_valid(hsv, mask):
    """Confere a cobertura de cor e evita grandes lacunas escuras."""
    x, y, w, h = cv2.boundingRect(mask)
    pixels = hsv[y:y+h, x:x+w][cv2.erode(mask[y:y+h, x:x+w], kernel(3)) > 0]
    if not len(pixels):
        return False
    hue, saturation, value = pixels.T
    dark_gaps = (saturation < 90) & (value < 150)
    warm = ((hue < 38) | (hue >= 170)) & (saturation >= 90)
    # Aceita apenas se a fração de falhas escuras for menor que 30%
    # e a fração de cores quentes for maior que 18%.
    return np.mean(dark_gaps) < .30 and np.mean(warm) > .18


def watershed_seeds(hsv_region, mask):
    """Seleciona as sementes vermelhas e verdes usadas pelo watershed."""
    # 1) Separa os canais H (matiz), S (saturação), V (brilho).
    h, s, v = cv2.split(hsv_region)
    # 2) SEMENTE VERMELHA: H perto de 0 ou 180, S>160, V>100.
    #    AND com a máscara garante que as sementes pertençam ao doce.
    red = (((h < 7) | (h > 173)) & (s > 160) & (v > 100) & (mask > 0)).astype(np.uint8)
    # 3) SEMENTE VERDE: H entre 38 e 84, S>110, V>45.
    green = ((h >= 38) & (h < 85) & (s > 110) & (v > 45) & (mask > 0)).astype(np.uint8)
    return red, green


# Calcula as cores predominantes de UM objeto usando seus pixels HSV.
# Retorna um dicionário, por exemplo {"verde": 0.7, "vermelho": 0.3}.
def color_distribution(hsv, object_mask, kind="minhoca"):
    x, y, w, h = cv2.boundingRect(object_mask)
    # Erosão remove parte das bordas, reduzindo contaminação pela cor do fundo.
    mask = cv2.erode(object_mask[y:y+h, x:x+w], kernel(3))
    pixels = hsv[y:y+h, x:x+w][mask > 0]
    if not len(pixels):
        return {"indefinida": 1.0}
    h, s, v = pixels.T
    # Regras de preto/cinza mudam nas ferramentas por causa do metal e cabo.
    tool = kind == "ferramenta"
    dark = ((v < 65) | ((s < 90) & (v < 150))) if tool else (v < 48)
    gray = ((s < 90) & (v >= 150)) if tool else ((v >= 48) & (s < 45))
    pink = ((h >= 145) | ((h <= 5) & (kind == "fita_rosa"))) & (s >= 45) & (s < 210) & (v >= 100)
    counts = {"preto": np.count_nonzero(dark), "cinza/prata": np.count_nonzero(gray),
              "rosa": np.count_nonzero(pink)}
    # Evita classificar duas vezes os pixels escuros, cinzas ou rosas.
    chromatic = (s >= (90 if tool else 45)) & ~dark & ~gray & ~pink
    bands = {"vermelho": (h < 10) | (h >= 170), "laranja": (h >= 10) & (h < 18),
             "amarelo": (h >= 18) & (h < 38), "verde": (h >= 38) & (h < 90),
             "azul": (h >= 90) & (h < 135), "roxo": (h >= 135) & (h < 170)}
    counts.update({name: int(np.count_nonzero(chromatic & band)) for name, band in bands.items()})
    # Calcula a participação de cada cor; exige no mínimo 8%
    # para ignorar pequenas manchas e reflexos.
    total = max(1, sum(counts.values()))
    result = {name: round(count / total, 3) for name, count in
              sorted(counts.items(), key=lambda item: -item[1]) if count / total >= .08}
    return result or {max(counts, key=counts.get): 1.0}


# Heurística para vídeo: busca grandes regiões HSV compatíveis com pele.
# Quando encontra provável mão, processing.py pausa temporariamente
# a atualização do rastreamento, para evitar entradas/saídas falsas.
# Não é um detector de mãos treinado; pode errar conforme a iluminação.
def hand_present(frame):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (9, 9), 0), cv2.COLOR_BGR2HSV)
    # Faixa HSV aproximada usada para marcar pixels com aspecto de pele.
    skin = cv2.inRange(hsv, np.array([0, 35, 100]), np.array([25, 145, 255]))
    # Remove pontos pequenos, preservando regiões maiores.
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel(9))
    h, w = skin.shape
    for contour in contours(skin):
        x, y, bw, bh = cv2.boundingRect(contour)
        fraction = cv2.contourArea(contour) / (h * w)
        # Dá mais peso a regiões que tocam a borda (mão entrando na cena).
        border = x < 3 or y < 3 or x + bw > w - 4 or y + bh > h - 4
        if (fraction > .012 and border) or fraction > .04:
            return True
    return False
