# Análise e Comparação de Técnicas de Esqueletização em Plântulas de Soja

Este repositório contém um estudo comparativo rigoroso de algoritmos de **esqueletização** e **afinamento morfológico** aplicados a imagens de segmentação de plântulas de soja (*Glycine max*), com foco na extração da estrutura morfológica e mensuração de órgãos vegetais (hipocótilo e raiz primária).

---

## 📌 Técnicas Avaliadas

O projeto implementa e analisa comparativamente quatro abordagens clássicas e modernas de extração de esqueleto:

1. **Zhang-Suen (1984) — Afinamento Puro:**
   * Algoritmo iterativo paralelo baseado em duas sub-iterações morfológicas de remoção de pixels de contorno que preservam a conectividade 8-vizinhos.
   * Rápido e amplamente utilizado, com tendência a preservar detalhes finos.

2. **Lee & Kashyap (1994) — Afinamento Topológico Puro:**
   * Algoritmo de afinamento baseado em número de Euler e conectividade topológica (implementado via `skimage.morphology.skeletonize(..., method='lee')`).
   * Produz esqueletos estritamente conexos e com espessura de 1 pixel.

3. **Transformada do Eixo Medial (MAT Puro) de Harry Blum (1967):**
   * Esqueletização geométrica baseada nos centros das esferas/círculos maximais inscritos no objeto, calculados via campo de distâncias (*Distance Transform*).
   * Sem podas, suscetível a ramificações espúrias causadas por irregularidades na borda.

4. **MAT com Poda Baseada em Distância (*Pruning*):**
   * Transformada do eixo medial enriquecida com filtragem morfológica e poda adaptativa de ramos espúrios fundamentada na função distância ao bordo, mantendo apenas a espinha dorsal biológica da plântula.

5. **Comparação Geral Consolidada:**
   * Notebook comparativo que executa as 4 abordagens lado a lado sobre os mesmos espécimes, gerando métricas visuais e quantitativas de preservação topológica e tempo de processamento.

---

## 📂 Estrutura do Projeto

```text
seedling-skeleton/
├── Comparacao_Tecnicas_Esqueletizacao.ipynb
├── Esqueletizacao_Lee_Afinamento_Puro.ipynb
├── Esqueletizacao_MAT_Puro.ipynb
├── Esqueletizacao_Zhang_Suen_Afinamento_Puro.ipynb
├── Medial_Axis_Transform_(MAT)_com_poda_baseada_em_distância.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

> **Nota:** As pastas de dados (`originals-dataset/` e `labeled-dataset/`) contêm arquivos de imagem binários e estão excluídas do controle de versão pelo `.gitignore`.

---

## 🗂️ Preparação do Dataset

Para executar os notebooks localmente, organize as imagens na raiz do projeto com as seguintes pastas:

* `originals-dataset/`: Diretório contendo as fotografias originais das plântulas (formato `.jpg`).
* `labeled-dataset/`: Diretório contendo as máscaras de segmentação anotadas correspondentes (formato `.png`).

Cada arquivo de máscara deve possuir correspondência de nome com a imagem original (por exemplo, `C10-R1_...png` correspondendo a `C10-R1_...jpg`).

---

## 🚀 Instalação e Execução

### 1. Clonar o Repositório
```bash
git clone https://github.com/<seu-usuario>/<nome-do-repositorio>.git
cd <nome-do-repositorio>
```

### 2. Criar e Ativar Ambiente Virtual
```bash
# Criar ambiente virtual
python3 -m venv .venv

# Ativar no Linux/macOS:
source .venv/bin/activate

# Ativar no Windows:
# .venv\Scripts\activate
```

### 3. Instalar Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Executar os Notebooks
Você pode iniciar os notebooks com Jupyter Lab, Jupyter Notebook ou diretamente pelo VS Code / PyCharm / Google Colab:
```bash
jupyter notebook
```

### 5. Executar a Interface Web Interativa (HTML/JS)
O projeto inclui um aplicativo web para teste rápido, seleção e upload de máscaras anotadas (`labeled-dataset/`) e comparação visual simultânea das 4 técnicas:
```bash
python3 server.py
```
Acesse no seu navegador: `http://localhost:8080`

---

## 🛠️ Tecnologias e Bibliotecas

* **Python 3.10+**
* [NumPy](https://numpy.org/) — Operações numéricas e manipulação matricial
* [OpenCV (`opencv-python`)](https://opencv.org/) — Carregamento, processamento de imagem e morfologia
* [Scikit-Image](https://scikit-image.org/) — Algoritmos de esqueletização (`skeletonize`, `medial_axis`)
* [SciPy](https://scipy.org/) — Transformada de distância (`distance_transform_edt`) e rotulagem de componentes
* [Matplotlib](https://matplotlib.org/) — Visualização dos gráficos, máscaras e esqueletos sobrepostos
