# TCU Busca IA 🏛️✨

Um sistema de busca avançado e offline para acórdãos do Tribunal de Contas da União (TCU). 
Este projeto combina buscas lexicais de alta performance (FTS5) com inteligência artificial para buscas semânticas (Embeddings) e sumarização generativa, garantindo resultados cirúrgicos mesmo em bases de dados com milhares de processos.

## 🚀 Funcionalidades

- **Busca Híbrida Inteligente**: Encontra acórdãos não apenas por palavras exatas, mas pelo significado e contexto jurídico (Busca Vetorial).
- **Sumarização com IA (Generativa)**: Geração de resumos sólidos e fluidos em parágrafos contínuos com 1 clique (consumindo a API OpenAI).
- **Extração Semântica Dinâmica**: O sistema varre o teor dos acórdãos localmente e sugere automaticamente os grandes temas em alta (ex: Licitação, Obras).
- **Filtros Avançados**: Filtre por ano, colegiado, tipo de processo e relator com amarração total no banco de dados.
- **100% Offline (Core)**: A busca vetorial e lexical rodam integralmente sem internet usando o banco SQLite local, poupando custos de cloud.
- **Performance**: Paginação infinita assíncrona para lidar fluidamente com bases gigantes.

## 🏗️ Arquitetura do Projeto

O sistema foi modularizado seguindo as melhores práticas do ecossistema FastAPI e Vanilla JS:

```
tcu-busca-ia/
├── data/                  # Base de acórdãos brutos (.md) e banco SQLite (ignorado no git)
├── src/                   # Módulos Python (Searcher, Indexer, Embedder)
├── static/                # Arquivos estáticos do Frontend
│   ├── css/style.css      # Estilos customizados
│   └── js/app.js          # Lógica assíncrona de renderização
├── index.html             # Esqueleto principal da Interface
├── main.py                # Ponto de entrada do FastAPI (Servidor Web)
├── cli.py                 # Ferramenta de linha de comando (Indexação offline)
├── requirements.txt       # Dependências do Python
└── .env.example           # Exemplo das variáveis de ambiente
```

## 🛠️ Tecnologias Utilizadas

### Backend
- **Python 3.10+**
- **FastAPI & Uvicorn**: Servidor assíncrono super veloz.
- **SQLite (FTS5)**: Banco de dados relacional e motor de busca em texto completo.
- **Sentence-Transformers**: Geração de embeddings jurídicos locais (`SamuelMauli/parity-embedding-juridico-br-v4`).
- **OpenAI SDK**: Para consumir LLMs generativas para sumarização.

### Frontend
- **HTML5 / Vanilla JS**: Aplicação Single Page Application (SPA) levíssima.
- **CSS3 Puro**: Design responsivo, moderno e totalmente isolado na pasta `static/css`.

## ⚙️ Como Executar o Projeto Localmente

### 1. Clonar o Repositório
```bash
git clone https://github.com/SEU_USUARIO/tcu-busca-ia.git
cd tcu-busca-ia
```

### 2. Criar o Ambiente Virtual e Instalar Dependências
```bash
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configurar a Chave de IA
Renomeie o arquivo `.env.example` para `.env` e insira a sua chave de API (OpenAI/TokenRouter) para habilitar o recurso de "Resumir com IA". O arquivo `.env` já está no `.gitignore` para a sua segurança.
```env
OPENAI_API_KEY=sk-sua_chave_aqui
```

### 4. Popular o Banco (Opcional)
Se precisar forçar a re-indexação ou popular os embeddings via terminal antes de ligar a web:
```bash
python cli.py --indexar
```

### 5. Iniciar o Servidor Web
```bash
python main.py
```
Acesse no navegador: [http://localhost:5000](http://localhost:5000)

---
*Desenvolvido com 💡 para inovar a pesquisa jurídica.*
