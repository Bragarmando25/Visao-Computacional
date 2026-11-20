# DOCES E FERRAMENTAS - VISAO COMPUTACIONAL
Armando Braga, João Meneguel, Leandro de Freitas e Pedro Pedroso

# COMO USAR
1. Abra Iniciar.bat.
2. Selecione uma foto ou video.
3. Clique Processar. A previa aparece na interface durante o processamento.
4. Clique Abrir resultados para acessar as imagens/videos anotados.
O botao Parar encerra o processamento e salva a parte concluida do video.

# INSTALACAO 
Use Python 3.10 a 3.12 com Tcl/Tk (incluido na instalacao padrao do Python).
Na pasta do projeto:
  python -m venv .venv
  .venv\Scripts\python -m pip install -r requirements.txt
  .venv\Scripts\python main.py
Versoes fixadas no requirements.txt: OpenCV 4.10.0.84 e NumPy 1.26.4.

# COMANDOS
1. python main.py --entrada "foto.jpeg"
2. python main.py --entrada "video.mp4"
3. python main.py --entrada "foto.jpeg" --saida resultados
4. python main.py --entrada "video.mp4" --analise-fps 15
Substitua python por .venv\Scripts\python se o ambiente nao estiver ativado.

# RESULTADOS
Cada arquivo recebe uma pasta com seu nome em resultados/.
1. Fotos: resultado.jpg.
2. Videos: resultado.mp4, ultimo_quadro.jpg.
O video de saida nao tem audio e preserva os quadros e a duracao original.
A analise ocorre a aproximadamente 15 quadros/s. Entradas e saidas exigem 0,4 e 0,6 s
de persistencia sem oclusao. O primeiro quadro estabelece o inventario inicial.

# TAMANHOS E CORES
Cada objeto recebe uma classe, ID, cores predominantes e medidas em pixels.
Os pixels se referem a imagem original. O comprimento de doces curvos e
estimado ao longo do esqueleto. Ferramentas incluem cabo e haste visivel.
Comprimento e largura sao informados em pixels.

# ARQUIVOS DO PROJETO
1. main.py                interface e comandos de processamento
2. doces_ferramentas/     segmentacao, medidas, rastreamento e exportacao
3. config.json            parametros HSV, geometria e tempos
4. resultados/            imagens e videos processados
5. README.md              instrucoes de uso

# FUNCAO DOS MODULOS
1. gui.py: abre a janela, permite selecionar arquivos, iniciar/parar o
   processamento, acompanhar a previa e abrir os resultados.
2. vision.py: le as imagens e a configuracao. O Detector coordena a
   segmentacao e as medidas e devolve os objetos encontrados.
3. hsv.py: concentra a conversao BGR para HSV, a correcao de brilho,
   as mascaras por cor, as cores predominantes e as sementes coloridas
   usadas pelo watershed. Tambem procura regioes com aspecto de pele.
4. watershed.py: usa sementes vermelhas/verdes e uma superficie de distancia
   para tentar separar duas minhocas encostadas. Rejeita divisões pequenas.
5. segmentation.py: coordena as mascaras HSV, os filtros de tamanho/formato
   e as chamadas ao watershed; recupera as hastes das ferramentas.
6. mask_utils.py: auxiliares compartilhados para criar elementos
   morfologicos, encontrar contornos e preencher mascaras.
7. measurements.py: calcula caixas e medidas em pixels, incluindo o
   comprimento curvo dos doces, e solicita as cores ao hsv.py. Reaproveita
   medidas estaveis durante o processamento dos videos.
8. tracking.py: acompanha os IDs entre quadros e confirma entradas/saidas
   quando a mudanca persiste, pausando durante a passagem da mao.
9. render.py: desenha as caixas, os IDs, as cores, as medidas e os contadores
   sobre a imagem ou o quadro do video.
10. processing.py: coordena a leitura, a analise e a gravacao. Fotos geram
   somente resultado.jpg; videos geram resultado.mp4 e ultimo_quadro.jpg.

# LIMITACOES
Os parametros atendem as fitas, minhocas e chaves amarelas/pretas fornecidas,
sobre mesa escura ou papel claro. Luz extrema, perspectiva, sobreposicoes e
objetos de aparencia semelhante podem exigir novos ajustes. Objetos cortados
pela borda sao medidos apenas na parte visivel. Durante a passagem da mao,
o inventario fica pausado.
A separacao de minhocas encostadas exige evidencia suficiente de sementes
vermelhas e verdes. Objetos completamente sobrepostos ou unioes sem essa
evidencia ainda podem ser contados como um unico objeto.
Fitas unidas em V podem formar uma regiao larga e deixar de ser reconhecidas,
afetando a contagem inicial e os eventos nos videos.
