"""Watershed com sementes de cor para separar minhocas encostadas.

Entrada: máscara candidata, imagem HSV, área da imagem e fração mínima.
Saída: duas máscaras válidas ou a máscara original quando falta evidência.
"""
import cv2
import numpy as np

from .hsv import watershed_seeds
from .mask_utils import filled


# ====================================================================
# WATERSHED: separar DUAS MINHOCAS ENCOSTADAS (VERMELHA/VERDE)
# ====================================================================
# Problema: a máscara colorida pode unir dois doces num único bloco.
# Watershed (divisor de águas) usa "sementes" de regiões conhecidas
# e expande seus rótulos sobre uma superfície, formando fronteiras.
# Imagine chuva caindo em duas bacias: a linha onde elas se encontram
# é a divisória entre os dois objetos.
# NESTE CÓDIGO a separação depende de DOIS conjuntos de pixels
# identificáveis: um vermelho e outro verde. Não é uma divisão
# genérica de todos os objetos encostados; mesma cor pode falhar.
# O resultado contém somente as regiões de marcadores 2 e 3.
def split_touching(mask, hsv, image_area, min_area_fraction):
    """Watershed: tenta separar minhocas vermelhas e verdes encostadas."""
    # 1) Guarda as dimensões originais para devolver máscaras completas.
    original_shape = mask.shape
    # 2) Recorta a região candidata com margem de 2 pixels para
    #    executar Watershed num pedaço pequeno, com menor custo.
    x, y, w, height = cv2.boundingRect(mask)
    x, y = max(0, x - 2), max(0, y - 2)
    w, height = min(w + 4, mask.shape[1] - x), min(height + 4, mask.shape[0] - y)
    # 3) Obtém a máscara local; branco=possível doce, preto=fundo.
    cropped = mask[y:y+height, x:x+w]
    # 4) Obtém as sementes de cor definidas em hsv.py.
    red, green = watershed_seeds(hsv[y:y+height, x:x+w], cropped)
    # 5) Cada cor precisa ter ao menos max(60 pixels, 3% da região).
    #    Evita dividir uma minhoca por reflexos/manchas minúsculos.
    minimum = max(60, np.count_nonzero(cropped) * .03)
    # 6) Sem vermelho E verde suficientes, NÃO tenta Watershed:
    #    devolve a região original como uma só minhoca.
    if red.sum() < minimum or green.sum() < minimum:
        return [mask]
    # 7) Define as etiquetas das sementes para o OpenCV:
    #    marcador 0 = pixels ainda desconhecidos;
    #    marcador 1 = fundo, isto é, fora da região colorida;
    #    marcador 2 = semente da minhoca vermelha;
    #    marcador 3 = semente da minhoca verde.
    #    Watershed modificará estes valores e marcará fronteiras com -1.
    markers = np.zeros_like(cropped, np.int32)
    markers[cropped == 0], markers[red > 0], markers[green > 0] = 1, 2, 3
    # 8) TRANSFORMADA DE DISTÂNCIA: cada pixel branco recebe
    #     sua distância euclidiana aproximada até o fundo preto.
    #     No interior espesso, a distância é maior; perto da borda, menor.
    #     DIST_L2 = distância euclidiana; 5 = tamanho da máscara de cálculo.
    distance = cv2.distanceTransform(cropped, cv2.DIST_L2, 5)
    # 9) Normaliza essas distâncias para 0..255 e INVERTE a escala.
    #     Assim, centros espessos ficam com valor baixo ("vales") e
    #     bordas ficam altas. Essa é a topografia para o Watershed.
    surface = np.uint8(255 - cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX))
    # 10) Executa Watershed. A superfície cinza é convertida em BGR,
    #     formato de 3 canais exigido pela função do OpenCV.
    #     O algoritmo EXPANDE as sementes e altera "markers" no lugar.
    #     Pixels de encontro entre regiões recebem rótulo -1 (fronteira).
    cv2.watershed(cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR), markers)
    # 11) Extrai duas máscaras: só os pixels atribuídos a 2 (vermelho)
    #     e a 3 (verde); fronteiras (-1) não pertencem a nenhuma.
    parts = [(markers == label).astype(np.uint8) * 255 for label in (2, 3)]
    # 12) Confere se AS DUAS regiões têm tamanho mínimo suficiente.
    #     Usa a área da imagem de análise como referência, não do recorte.
    if not all(np.count_nonzero(p) > image_area * min_area_fraction for p in parts):
        return [mask]
    # 13) Recria máscaras no tamanho original e preenche contornos.
    #     Devolve duas regiões que serão medidas separadamente.
    result = []
    for part in parts:
        full = np.zeros(original_shape, np.uint8)
        full[y:y+height, x:x+w] = filled(part)
        result.append(full)
    return result
