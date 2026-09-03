"""
Buscador Semântico de Acórdãos do TCU
Backend Flask com pré-busca textual + LLM via TokenRouter
"""

import os
import re
import json
import unicodedata
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI

# ──────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────
API_KEY = os.environ.get("TOKENROUTER_API_KEY", "sk-s82a7BxTCZSq9SWZZwnBZnESPTOfNgvSJsLNZCtAXLt4jzdq")
BASE_URL = "https://api.tokenrouter.com/v1"
MODEL = "z-ai/glm-5.3-free"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ──────────────────────────────────────────────
# Parser do Markdown
# ──────────────────────────────────────────────

def parse_acordaos(filepath):
    """Parseia o arquivo .md e retorna lista de dicts com os campos de cada acórdão."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Divide por '---' que separa cada acórdão
    blocos = re.split(r"\n---\n", content)
    acordaos = []

    for bloco in blocos:
        bloco = bloco.strip()
        if not bloco or bloco.startswith("# Jurisprudência"):
            continue

        acordao = {}

        # Título da seção (## ...)
        titulo_match = re.search(r"^##\s+(.+)$", bloco, re.MULTILINE)
        acordao["titulo"] = titulo_match.group(1).strip() if titulo_match else ""

        # Campos com padrão **Campo:** Valor
        campos = {
            "key": r"\*\*Chave \(KEY\):\*\*\s*(.+)",
            "tipo": r"\*\*Tipo:\*\*\s*(.+)",
            "numero": r"\*\*Número do Acórdão:\*\*\s*(\d+)",
            "ano": r"\*\*Ano:\*\*\s*(\d+)",
            "colegiado": r"\*\*Colegiado:\*\*\s*(.+)",
            "relator": r"\*\*Relator:\*\*\s*(.+)",
            "tipo_processo": r"\*\*Tipo de Processo:\*\*\s*(.+)",
            "entidade": r"\*\*Entidade:\*\*\s*(.+)",
            "assunto": r"\*\*Assunto:\*\*\s*(.+)",
            "sumario": r"\*\*Sumário:\*\*\s*(.+)",
        }

        for campo, pattern in campos.items():
            match = re.search(pattern, bloco)
            acordao[campo] = match.group(1).strip() if match else ""

        # Decisão — tudo após "**Acórdão (Decisão):**"
        decisao_match = re.search(r"\*\*Acórdão \(Decisão\):\*\*\s*\n(.+)", bloco, re.DOTALL)
        acordao["decisao"] = decisao_match.group(1).strip() if decisao_match else ""

        # Só adiciona se tem chave válida
        if acordao.get("key"):
            acordaos.append(acordao)

    return acordaos


# ──────────────────────────────────────────────
# Pré-busca textual (keyword matching)
# ──────────────────────────────────────────────

STOPWORDS_PT = {
    "a", "o", "e", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "um", "uma", "uns", "umas", "por", "para", "com", "sem", "sob", "sobre",
    "que", "se", "ao", "aos", "à", "às", "ou", "é", "são", "foi", "ser", "ter",
    "como", "mais", "menos", "muito", "entre", "pelo", "pela", "pelos", "pelas",
    "este", "esta", "esse", "essa", "isso", "isto", "aquele", "aquela",
    "seu", "sua", "seus", "suas", "meu", "minha", "nosso", "nossa",
    "todo", "toda", "todos", "todas", "outro", "outra", "outros", "outras",
    "qual", "quais", "quando", "onde", "quem", "já", "ainda", "também",
    "não", "sim", "nem", "mas", "porém", "contudo", "então", "assim",
    "lhe", "lhes", "me", "te", "nos", "vos", "os", "as",
}


def normalize_text(text):
    """Remove acentos e converte para minúsculas."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def tokenize(text):
    """Tokeniza e remove stopwords."""
    normalized = normalize_text(text)
    tokens = re.findall(r"\b[a-z]{3,}\b", normalized)
    return [t for t in tokens if t not in {normalize_text(s) for s in STOPWORDS_PT}]


def keyword_score(acordao, query_tokens):
    """Calcula score de relevância baseado em keywords."""
    score = 0
    campos_peso = {
        "sumario": 3,
        "assunto": 3,
        "entidade": 2,
        "titulo": 2,
        "tipo_processo": 1,
        "decisao": 1,
    }

    for campo, peso in campos_peso.items():
        texto = normalize_text(acordao.get(campo, ""))
        for token in query_tokens:
            if token in texto:
                score += peso
                # Bonus por match exato de palavra
                if re.search(r"\b" + re.escape(token) + r"\b", texto):
                    score += peso

    return score


def pre_busca(acordaos, query, top_n=30):
    """Retorna os top_n acórdãos mais relevantes por keyword matching."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return acordaos[:top_n]

    scored = []
    for ac in acordaos:
        sc = keyword_score(ac, query_tokens)
        if sc > 0:
            scored.append((sc, ac))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ac for _, ac in scored[:top_n]]


# ──────────────────────────────────────────────
# Chamada à LLM
# ──────────────────────────────────────────────

def chamar_llm(query, candidatos):
    """Envia candidatos para a LLM ranquear semanticamente."""
    # Monta resumo dos candidatos para o prompt
    candidatos_texto = ""
    for i, ac in enumerate(candidatos):
        candidatos_texto += f"""
[{i+1}] KEY: {ac['key']}
Título: {ac['titulo']}
Colegiado: {ac['colegiado']} | Relator: {ac['relator']}
Tipo: {ac['tipo_processo']} | Entidade: {ac['entidade']}
Assunto: {ac['assunto']}
Sumário: {ac['sumario'][:500]}
---
"""

    system_prompt = """Você é um assistente jurídico especializado em jurisprudência do Tribunal de Contas da União (TCU).
Sua tarefa é analisar uma lista de acórdãos candidatos e ranqueá-los por relevância semântica em relação à busca do usuário.

REGRAS:
1. Retorne APENAS um JSON válido, sem markdown, sem explicações adicionais fora do JSON.
2. Selecione os 10 mais relevantes (ou menos, se não houver 10 relevantes).
3. Para cada resultado, forneça: key, titulo, relevancia (1-10), resumo (2 linhas explicando POR QUE é relevante para a busca).
4. Ordene do mais relevante para o menos relevante.
5. Se nenhum candidato for relevante, retorne uma lista vazia.

FORMATO DE RESPOSTA (JSON puro):
{"resultados": [{"key": "...", "titulo": "...", "relevancia": 9, "resumo": "..."}, ...]}"""

    user_prompt = f"""BUSCA DO USUÁRIO: "{query}"

ACÓRDÃOS CANDIDATOS:
{candidatos_texto}

Analise e retorne o JSON com os resultados ranqueados."""

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            stream_options={"include_usage": True},
            extra_body={},
        )

        content_parts = []
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content_parts.append(delta.content)

        full_content = "".join(content_parts).strip()

        # Tenta extrair JSON de dentro de blocos de código, se houver
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", full_content)
        if json_match:
            full_content = json_match.group(1).strip()

        # Remove possíveis caracteres antes/depois do JSON
        start = full_content.find("{")
        end = full_content.rfind("}") + 1
        if start != -1 and end > start:
            full_content = full_content[start:end]

        resultado = json.loads(full_content)
        return resultado.get("resultados", [])

    except Exception as e:
        print(f"[LLM Error] {e}")
        # Fallback: retorna os candidatos sem ranqueamento da LLM
        fallback = []
        for ac in candidatos[:10]:
            fallback.append({
                "key": ac["key"],
                "titulo": ac["titulo"],
                "relevancia": 5,
                "resumo": ac.get("assunto", "Sem resumo disponível."),
            })
        return fallback


# ──────────────────────────────────────────────
# Carrega acórdãos em memória
# ──────────────────────────────────────────────
# Carrega acórdãos em memória
# ──────────────────────────────────────────────
print(f"[INFO] Buscando acórdãos no diretório {DATA_DIR}...")
ACORDAOS = []

if os.path.exists(DATA_DIR):
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".md"):
            filepath = os.path.join(DATA_DIR, filename)
            print(f"[INFO] Processando {filename}...")
            ACORDAOS.extend(parse_acordaos(filepath))
else:
    print(f"[WARNING] Diretório {DATA_DIR} não encontrado. Crie a pasta e coloque seus arquivos .md dentro.")

print(f"[INFO] {len(ACORDAOS)} acórdãos carregados com sucesso.")

# Index por key para acesso rápido
ACORDAOS_INDEX = {ac["key"]: ac for ac in ACORDAOS}

# Extrai valores únicos para filtros
COLEGIADOS = sorted(set(ac["colegiado"] for ac in ACORDAOS if ac["colegiado"]))
TIPOS_PROCESSO = sorted(set(ac["tipo_processo"] for ac in ACORDAOS if ac["tipo_processo"]))
RELATORES = sorted(set(ac["relator"] for ac in ACORDAOS if ac["relator"]))
ANOS = sorted(set(ac["ano"] for ac in ACORDAOS if ac["ano"]), reverse=True)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/filtros")
def filtros():
    """Retorna opções de filtros disponíveis."""
    return jsonify({
        "colegiados": COLEGIADOS,
        "tipos_processo": TIPOS_PROCESSO,
        "relatores": RELATORES,
        "anos": ANOS,
        "total_acordaos": len(ACORDAOS),
    })


@app.route("/api/buscar", methods=["POST"])
def buscar():
    """Endpoint principal de busca semântica."""
    data = request.get_json()
    query = data.get("query", "").strip()
    filtro_colegiado = data.get("colegiado", "")
    filtro_tipo = data.get("tipo_processo", "")
    filtro_relator = data.get("relator", "")
    filtro_ano = data.get("ano", "")

    if not query:
        return jsonify({"error": "Query não pode ser vazia"}), 400

    # Aplica filtros primeiro
    base = ACORDAOS
    print(f"[DEBUG] query: {query}, filtros: ano={filtro_ano}, colegiado={filtro_colegiado}, tipo={filtro_tipo}")
    print(f"[DEBUG] Tamanho inicial de ACORDAOS: {len(ACORDAOS)}")
    if filtro_ano:
        try:
            filtro_ano_int = int(filtro_ano)
            base = [ac for ac in base if ac["ano"] == filtro_ano_int]
        except ValueError:
            pass
    if filtro_colegiado:
        base = [ac for ac in base if ac["colegiado"] == filtro_colegiado]
    if filtro_tipo:
        base = [ac for ac in base if ac["tipo_processo"] == filtro_tipo]
    if filtro_relator:
        base = [ac for ac in base if ac["relator"] == filtro_relator]

    print(f"[DEBUG] Tamanho da base apos filtros: {len(base)}")
    # Pré-busca textual
    candidatos = pre_busca(base, query, top_n=30)
    print(f"[DEBUG] Tamanho de candidatos apos pre-busca: {len(candidatos)}")

    if not candidatos:
        return jsonify({"resultados": [], "total_candidatos": 0, "mensagem": "Nenhum acórdão encontrado para esta busca."})

    # Envia para LLM ranquear
    resultados = chamar_llm(query, candidatos)

    # Enriquece resultados com dados completos
    resultados_enriquecidos = []
    for r in resultados:
        ac = ACORDAOS_INDEX.get(r.get("key", ""))
        if ac:
            resultados_enriquecidos.append({
                **r,
                "numero": ac.get("numero", ""),
                "ano": ac.get("ano", ""),
                "colegiado": ac.get("colegiado", ""),
                "relator": ac.get("relator", ""),
                "tipo_processo": ac.get("tipo_processo", ""),
                "entidade": ac.get("entidade", ""),
                "assunto": ac.get("assunto", ""),
            })

    return jsonify({
        "resultados": resultados_enriquecidos,
        "total_candidatos": len(candidatos),
        "query": query,
    })


@app.route("/api/acordao/<key>")
def acordao_detalhe(key):
    """Retorna o acórdão completo por chave."""
    ac = ACORDAOS_INDEX.get(key)
    if not ac:
        return jsonify({"error": "Acórdão não encontrado"}), 404
    return jsonify(ac)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Servidor iniciando em http://localhost:5000")
    app.run(debug=True, port=5000)
