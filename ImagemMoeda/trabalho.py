import os
import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread("moedas.jpeg")

if imagem is None:
    raise FileNotFoundError("Imagem não encontrada!")

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

limiares = (60, 120)
binarias = {
    limiar: cv2.threshold(cinza, limiar, 255, cv2.THRESH_BINARY)[1]
    for limiar in limiares
}

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

erosoes = {
    limiar: cv2.erode(imagem_binaria, kernel, iterations=1)
    for limiar, imagem_binaria in binarias.items()
}

dilatacoes = {
    limiar: cv2.dilate(imagem_binaria, kernel, iterations=1)
    for limiar, imagem_binaria in binarias.items()
}

os.makedirs("resultados", exist_ok=True)
cv2.imwrite("resultados/escala_cinza.jpg", cinza)

for limiar in limiares:
    cv2.imwrite(f"resultados/binaria_{limiar}.jpg", binarias[limiar])
    cv2.imwrite(f"resultados/erosao_{limiar}.jpg", erosoes[limiar])
    cv2.imwrite(f"resultados/dilatacao_{limiar}.jpg", dilatacoes[limiar])

print("Imagens salvas com sucesso!")

def exibir_resultados(imagens, titulo_figura, titulo_operacao):
    figura, eixos = plt.subplots(1, len(limiares), figsize=(15, 5))
    figura.suptitle(titulo_figura)

    for eixo, limiar in zip(eixos, limiares):
        eixo.imshow(imagens[limiar], cmap="gray")
        eixo.set_title(f"{titulo_operacao} - T = {limiar}")
        eixo.axis("off")

    figura.tight_layout()
    plt.show(block=True)


exibir_resultados(binarias, "Imagens binárias", "Binária")
exibir_resultados(erosoes, "Erosões", "Erosão")
exibir_resultados(dilatacoes, "Dilatações", "Dilatação")
