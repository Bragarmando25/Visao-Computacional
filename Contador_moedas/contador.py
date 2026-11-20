import cv2
import numpy as np

#Abrir a imagem
imagem = cv2.imread("objetos.png")

if imagem is None:
    print("Erro: não foi possível abrir a imagem.")
    exit()

#Converter para escala de cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

#Binarização automática usando Otsu

_, binaria = cv2.threshold(
    cinza,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

#Limpeza da imagem com abertura morfológica

kernel = np.ones((3, 3), np.uint8)

abertura = cv2.morphologyEx(
    binaria,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)

#Obter a região de fundo garantido

fundo_certo = cv2.dilate(
    abertura,
    kernel,
    iterations=3
)

# Transformada de distância, ela mede a distância de cada pixel branco até o fundo.
# Como as moedas são maiores que os grãos de arroz, o centro das moedas possui uma distância muito maior.

distancia = cv2.distanceTransform(
    abertura,
    cv2.DIST_L2,
    5
)


#Criar o primeiro plano garantido
# O valor 0.55 foi adequado para esta imagem:
# mantém os centros das 5 moedas e elimina os grãos de arroz.

_, frente_certa = cv2.threshold(
    distancia,
    0.55 * distancia.max(),
    255,
    0
)

frente_certa = np.uint8(frente_certa)

#Região desconhecida

desconhecida = cv2.subtract(
    fundo_certo,
    frente_certa
)

#Criar marcadores com connectedComponents

quantidade_rotulos, marcadores = cv2.connectedComponents(
    frente_certa
)

# Soma 1 para que o fundo seja o rótulo 1, o valor 0 ficará reservado para a região desconhecida.
marcadores = marcadores + 1

marcadores[desconhecida == 255] = 0

#Aplicar Watershed
# O algoritmo encontra as fronteiras entre as moedas
# que estão encostadas.

marcadores = cv2.watershed(
    imagem.copy(),
    marcadores
)


#Encontrar contornos e centros das moedas

resultado = imagem.copy()

moedas = []

# Depois do watershed:
# -1 = fronteira
#  1 = fundo
#  2, 3, 4... = objetos/moedas
for rotulo in range(2, quantidade_rotulos + 1):

    mascara = np.uint8(
        marcadores == rotulo
    ) * 255

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contornos) == 0:
        continue

    # Usa o maior contorno encontrado
    contorno = max(
        contornos,
        key=cv2.contourArea
    )

    # Calcula o centro da moeda
    M = cv2.moments(contorno)

    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    moedas.append(
        (cy, cx, contorno)
    )

# Organiza visualmente os objetos para numerar
# de cima para baixo e da esquerda para a direita.
moedas.sort(
    key=lambda item: (
        item[0] // 120,
        item[1]
    )
)


#Desenhar contornos e números

for numero, (_, _, contorno) in enumerate(
    moedas,
    start=1
):
    M = cv2.moments(contorno)

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Contorno verde
    cv2.drawContours(
        resultado,
        [contorno],
        -1,
        (0, 255, 0),
        3
    )

    # Número da moeda em vermelho
    texto = str(numero)

    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.9
    espessura = 2

    (largura, altura), _ = cv2.getTextSize(
        texto,
        fonte,
        escala,
        espessura
    )

    cv2.putText(
        resultado,
        texto,
        (
            cx - largura // 2,
            cy + altura // 2
        ),
        fonte,
        escala,
        (0, 0, 255),
        espessura,
        cv2.LINE_AA
    )


#Mostrar o total na imagem e no terminal

total = len(moedas)

print("Total de moedas:", total)

cv2.putText(
    resultado,
    f"Total de moedas: {total}",
    (15, 32),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

#Salvar resultado

cv2.imwrite(
    "resultado_moedas.png",
    resultado
)

# Mostrar a imagem na tela
cv2.imshow(
    "Resultado",
    resultado
)

cv2.waitKey(0)
cv2.destroyAllWindows()
