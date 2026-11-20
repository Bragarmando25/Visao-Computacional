"""Entrada do detector: leitura de arquivos, configuração e coordenação das etapas.

Os algoritmos ficam em hsv.py e watershed.py; a ordem da
segmentação fica em segmentation.py, e as medidas em measurements.py.
"""
from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np

from .measurements import Detection, LABELS, Measurer
from .segmentation import Segmenter
from .hsv import hand_present

ROOT = Path(__file__).resolve().parents[1]


# Lê os limites HSV, áreas mínimas e demais parâmetros do config.json.
# Se nenhum caminho for passado, procura o arquivo na raiz do projeto.
def load_config(path=None):
    return json.loads(Path(path or ROOT / "config.json").read_text(encoding="utf-8"))


# Abre uma foto como matriz NumPy em BGR. imdecode + fromfile
# permite ler arquivos cujo caminho contém acentos no Windows.
def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    # Se a leitura falhou, interrompe com uma mensagem útil.
    if image is None:
        raise ValueError(f"Não foi possível ler a imagem: {path}")
    return image


# Grava a imagem de saída. Cria a pasta de destino quando necessário.
def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Codifica a matriz no formato indicado pela extensão (.jpg, .png...).
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Não foi possível codificar {path}")
    encoded.tofile(str(path))


class Detector:
    """Recebe uma imagem e devolve os objetos com classes, cores e medidas."""

    def __init__(self, config=None):
        self.config = config or load_config()
        self._segmenter = Segmenter(self.config)
        self._measurer = Measurer()

    def detect(self, original):
        # Reduz a imagem apenas para análise; as medidas voltam à escala original.
        height, width = original.shape[:2]
        scale = min(1.0, self.config["max_dimension"] / max(height, width))
        image = cv2.resize(original, None, fx=scale, fy=scale) if scale < 1 else original.copy()
        objects, hsv = self._segmenter.segment(image)
        return self._measurer.measure(objects, hsv, scale, image.shape[1])
