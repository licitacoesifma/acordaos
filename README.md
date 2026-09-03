# Sistema de Busca de Acórdãos TCU

Um motor de busca avançado projetado para indexar e pesquisar decisões (acórdãos) do Tribunal de Contas da União (TCU). O sistema realiza a extração de dados brutos em Markdown e constrói uma API ultrarrápida combinando buscas por palavra-chave e compreensão semântica, proporcionando resultados altamente relevantes e precisos.

## 🚀 Arquitetura e Tecnologias

- **Motor Lexical (Palavras-chave)**: SQLite com **FTS5** e ranking BM25, utilizando tokenizador customizado com remoção de diacríticos para ignorar acentos.
- **Motor Semântico (Compreensão de Contexto)**: Embeddings vetoriais locais baseados no modelo jurídico brasileiro `SamuelMauli/parity-embedding-juridico-br-v4` (~117M parâmetros).
- **Busca Híbrida**: Algoritmo **RRF (Reciprocal Rank Fusion)** para fundir os resultados lexicais e semânticos.
- **API Backend**: Construída com **FastAPI**, servindo endpoints REST documentados via Swagger.
- **Frontend**: Interface web dinâmica (`index.html`) conectada diretamente à API.

## 📂 Estrutura do Projeto

```text
acordaos/
├── data/
│   └── acordao-completo-2026.md    # Base bruta de dados do TCU (4.566 acórdãos)
├── db/
│   └── acordaos.db                 # Banco de dados SQLite gerado
├── src/
│   ├── chunker.py                  # Parser e extrator de entidades do Markdown
│   ├── indexer.py                  # Gestão do banco de dados e FTS5
│   ├── embedder.py                 # Integração com o SentenceTransformers
│   └── searcher.py                 # Lógica da busca Híbrida/Lexical/Semântica
├── main.py                         # CLI para indexação e testes rápidos
├── server.py                       # Servidor web FastAPI
├── requirements.txt                # Dependências Python
└── index.html                      # Interface do usuário (Frontend)
```

## 🛠️ Como Usar

### 1. Instalar Dependências

Certifique-se de usar Python 3.9+ e execute:

```bash
pip install -r requirements.txt
```

### 2. Indexar a Base de Dados

Antes da primeira busca, é necessário extrair o arquivo Markdown, criar o banco e gerar os embeddings vetoriais (processo executado apenas uma vez e que pode demorar alguns minutos dependendo do hardware):

```bash
python main.py --indexar
```

### 3. Rodar a Aplicação

Inicie o servidor local FastAPI:

```bash
python server.py
```

Acesse no navegador:
- **Interface Web:** [http://localhost:5000](http://localhost:5000)
- **Documentação da API (Swagger):** [http://localhost:5000/docs](http://localhost:5000/docs)

### 4. Usar a Interface de Linha de Comando (Opcional)

Você pode realizar buscas rápidas sem iniciar o servidor:

```bash
python main.py --buscar "fraude previdenciária" --modo hibrida -k 5
python main.py --buscar "licitação" --modo lexical
```

## 📊 Endpoints da API

- `GET /` — Interface Web
- `POST /api/buscar` — Executa uma busca (Lexical, Semântica ou Híbrida).
- `GET /api/acordao/{chave}` — Retorna o detalhamento completo de uma decisão.
- `GET /api/filtros` — Retorna os filtros disponíveis (anos, relatores, etc).
- `GET /api/stats` — Retorna estatísticas de indexação.
