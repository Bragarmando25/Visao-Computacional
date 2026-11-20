<div align="center">👁️ Visão Computacional
Projetos acadêmicos de processamento e análise de imagens
Repositório com projetos desenvolvidos durante os estudos de Visão Computacional no curso de Ciência da Computação da Universidade Estadual do Norte do Paraná (UENP).
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Pillow-306998?style=for-the-badge" alt="Pillow" />
</p>
Meu GitHub · LinkedIn
</div>📖 Sobre o repositório
Esta coleção reúne atividades e projetos práticos voltados à interpretação de imagens e vídeos por meio de algoritmos de visão computacional.

Os trabalhos exploram o uso de Python e OpenCV para manipulação de imagens, análise de cores, identificação de objetos e extração de informações visuais. Cada pasta contém um projeto ou conjunto de arquivos relacionado a um exercício específico.
📂 Projetos
Pasta	Descrição
Contador_moedas	Projeto dedicado à contagem de moedas.
ImagemMoeda	Arquivos relacionados ao trabalho com imagens de moedas.
Doces	Sistema de identificação, classificação, medição e contagem de doces e ferramentas em imagens e vídeos.


Nota: As descrições das pastas de moedas são intencionalmente gerais. Consulte os arquivos de cada pasta para conhecer sua implementação específica.

🔎 Projeto em destaque — Doces e Ferramentas
O projeto Doces reúne técnicas clássicas de visão computacional para identificar objetos com características visuais semelhantes e acompanhar mudanças ao longo de um vídeo.

Objetos trabalhados: fitas doces rosas, minhocas de gelatina coloridas e ferramentas com cabos amarelos e pretos.
Funcionalidades
- Detecção e classificação de objetos a partir de características de cor e formato.
- Segmentação por HSV, com máscaras de cores e operações morfológicas.
- Análise de contornos e medidas de comprimento e largura em pixels.
- Watershed para auxiliar na separação de objetos encostados.
- Rastreamento em vídeo, com IDs e confirmação de entradas e saídas.
- Interface gráfica para selecionar arquivos, acompanhar o processamento e abrir os resultados.
- Exportação de imagens e vídeos anotados com informações da detecção.
Etapas gerais de processamento
Imagem ou vídeo
      ↓
Pré-processamento
      ↓
Segmentação por cores (HSV)
      ↓
Morfologia e análise de contornos
      ↓
Separação e classificação dos objetos
      ↓
Medições e rastreamento (em vídeos)
      ↓
Visualização e exportação dos resultados
Limitações conhecidas: o reconhecimento depende das condições de iluminação, da perspectiva e do grau de sobreposição entre objetos. As medidas são expressas em pixels, não em centímetros.
Desenvolvimento em equipe
Este projeto foi desenvolvido por Armando Braga, João Meneguel, Leandro de Freitas e Pedro Pedroso.
🧰 Tecnologias e conceitos
Categoria	Recursos empregados
Linguagem	Python
Visão computacional	OpenCV
Operações numéricas	NumPy
Imagens	Pillow
Técnicas de processamento	HSV, máscaras, morfologia, contornos e Watershed
Análise temporal	Rastreamento de objetos e contagem de entradas/saídas


A lista reúne recursos documentados no projeto Doces. Os demais projetos podem utilizar dependências e técnicas diferentes.

▶️ Como executar
Os requisitos e a forma de execução podem variar entre as pastas. Para o sistema Doces, a documentação fornecida recomenda Python 3.10 a 3.12, com Tcl/Tk instalado.

Caso os arquivos main.py e requirements.txt estejam diretamente dentro de Doces/, no Windows:
git clone https://github.com/Bragarmando25/Visao-Computacional.git
cd Visao-Computacional\Doces
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python main.py
Também é possível processar arquivos pelo terminal:
.\.venv\Scripts\python main.py --entrada "caminho/para/imagem.jpg"
.\.venv\Scripts\python main.py --entrada "caminho/para/video.mp4"
O projeto permite configurar parâmetros de detecção por meio de config.json. Para os demais trabalhos, consulte os arquivos e as instruções de cada pasta.
🎓 Contexto acadêmico
Trabalhos desenvolvidos como parte da formação em Ciência da Computação na UENP, com foco na aplicação prática de conceitos de processamento digital de imagens e visão computacional.
👨‍💻 Autor
Armando Braga
Ciência da Computação — UENP
GitHub · LinkedIn
<div align="center">
  <sub>Repositório acadêmico para estudo, experimentação e demonstração de projetos de Visão Computacional.</sub>
</div>
