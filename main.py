"""
API FastAPI para Busca de Acordaos do TCU.

Endpoints:
    GET  /                     -> Serve o index.html
    POST /api/buscar           -> Busca lexical / semantica / hibrida
    POST /api/resumir          -> Resume um acordao via LLM
    GET  /api/filtros          -> Opcoes de filtros e temas em alta
    GET  /api/stats            -> Estatisticas do banco de dados
    GET  /api/acordao/{chave}  -> Retorna um acordao completo

Uso:
    python main.py
    # ou
    uvicorn main:app --reload --port 5000
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError

from src.indexer import get_connection, contar_acordaos, inserir_acordaos
from src.searcher import (
    busca_lexical,
    busca_semantica,
    busca_hibrida,
    calcular_relevancia,
    invalidar_cache_embeddings,
)
from src.chunker import carregar_todos_acordaos
from src.embedder import gerar_embeddings

# Carrega variáveis de ambiente
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# Estado global da aplicacao
# ──────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()
_sync_in_progress = False
_last_data_signature = None

# Os endpoints agora rodam em threads do threadpool do FastAPI (ver `def`
# em vez de `async def` nas rotas abaixo). sqlite3.Connection nao e segura
# para uso concorrente pela MESMA instancia entre threads — por isso cada
# thread recebe sua propria conexao (thread-local) em vez de todas
# compartilharem uma unica conexao serializada por um lock global. Com o
# banco em modo WAL (ver get_connection), multiplas conexoes podem ler em
# paralelo sem se bloquear, entao isso remove o gargalo que forcava toda
# busca/filtro/stat a esperar em fila atras de um unico lock, mesmo sendo
# operacoes somente leitura.
_conn_local = threading.local()
_all_conns_lock = threading.Lock()
_all_conns: list = []

# Pool dedicado para as chamadas de LLM em /api/resumir. Precisamos dele porque
# `future.result(timeout=...)` e a UNICA forma de impor um prazo total real
# sobre uma chamada de streaming sincrona: o timeout do proprio cliente HTTP
# (OpenAI/httpx) so cobre cada leitura individual, entao um provedor que vai
# entregando poucos tokens de forma continua (sem nunca ficar muito tempo em
# silencio) nunca dispara aquele limite, mesmo que o total passe de minutos —
# foi exatamente o que aconteceu ao testar com o modelo gratuito padrao.
# max_workers alto o bastante para nao enfileirar resumos concorrentes atras
# de uma chamada lenta/pendurada (a thread abandonada apos o timeout do
# future continua rodando em segundo plano ate o proprio timeout do cliente
# HTTP encerra-la, ja que Python nao permite matar uma thread de fora).
_llm_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm-resumo")


def _assinatura_pasta_data(data_dir: str):
    """Assinatura barata (nome, tamanho, mtime) dos .md em data/.

    Usada para pular o reparsing caro dos Markdowns quando nada mudou
    desde a ultima sincronizacao (evita reprocessar dezenas de MB a
    cada acesso a pagina inicial).
    """
    if not os.path.isdir(data_dir):
        return None
    arquivos = []
    for nome in sorted(os.listdir(data_dir)):
        if nome.endswith(".md"):
            caminho = os.path.join(data_dir, nome)
            stat = os.stat(caminho)
            arquivos.append((nome, stat.st_size, stat.st_mtime))
    return tuple(arquivos)

def _get_model():
    """Carrega o modelo de embeddings (lazy loading, thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from src.embedder import carregar_modelo
                _model = carregar_modelo()
    return _model

def _get_conn():
    """Retorna a conexao com o banco de dados da thread atual."""
    conn = getattr(_conn_local, "conn", None)
    if conn is None:
        conn = get_connection()
        _conn_local.conn = conn
        with _all_conns_lock:
            _all_conns.append(conn)
    return conn

def _sync_novos_acordaos():
    """Tarefa em background para buscar novos arquivos em data/ e indexar."""
    global _sync_in_progress, _last_data_signature
    if _sync_in_progress:
        return

    _sync_in_progress = True
    try:
        DATA_DIR = os.path.join(BASE_DIR, "data")

        # Pula o reparsing caro (MDs de dezenas de MB) se nada mudou em
        # data/ desde a ultima sincronizacao: esta funcao roda a cada
        # acesso a "/", entao sem isso o servidor reprocessaria os
        # mesmos arquivos gigantes em toda visita a home.
        assinatura = _assinatura_pasta_data(DATA_DIR)
        if assinatura is not None and assinatura == _last_data_signature:
            return

        conn = _get_conn()

        # 1. Carrega todos os acórdãos dos arquivos MD
        acordaos = carregar_todos_acordaos(DATA_DIR)
        if not acordaos:
            _last_data_signature = assinatura
            return

        # 2. Insere no banco (INSERT OR IGNORE cuida para não duplicar)
        inseridos = inserir_acordaos(conn, acordaos)
        com_embedding = conn.execute("SELECT COUNT(*) FROM acordaos WHERE embedding IS NOT NULL").fetchone()[0]
        total = contar_acordaos(conn)

        # 3. Se houve novos inseridos ou há acórdãos sem embedding, atualiza
        if inseridos > 0 or com_embedding < total:
            model = _get_model()
            gerar_embeddings(conn, model)
            # Novos vetores no banco: invalida o cache em memoria da busca
            # semantica para que a proxima busca recarregue a matriz atualizada.
            invalidar_cache_embeddings()
            print(f"[INFO] Sincronização concluída. {total - com_embedding} novos embeddings gerados.")

        # So marca a assinatura como sincronizada apos o sucesso de todas as
        # etapas acima: se inserir_acordaos ou gerar_embeddings falharem, a
        # excecao abaixo impede esta linha e a proxima visita tenta de novo,
        # em vez de achar (incorretamente) que já sincronizou.
        _last_data_signature = assinatura

    except Exception as e:
        print(f"[ERRO] Falha na sincronização em background: {e}")
    finally:
        _sync_in_progress = False

# ──────────────────────────────────────────────
# Lifespan (startup/shutdown)
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicacao."""
    conn = _get_conn()
    total = contar_acordaos(conn)
    print(f"[INFO] Banco de dados carregado: {total} acordaos")

    if total == 0:
        print("[AVISO] Banco vazio. Execute 'python cli.py --indexar' primeiro.")

    yield

    # Shutdown: fecha todas as conexoes thread-local criadas durante a vida do app.
    with _all_conns_lock:
        for c in _all_conns:
            try:
                c.close()
            except Exception:
                pass
        _all_conns.clear()
    print("[INFO] Conexao com banco de dados encerrada.")

    # Nao espera chamadas de LLM pendentes terminarem (podem estar presas por
    # ate 30s no timeout do cliente HTTP) para nao atrasar o shutdown.
    _llm_executor.shutdown(wait=False, cancel_futures=True)


# ──────────────────────────────────────────────
# Criacao do app FastAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="Buscador de Acordaos TCU",
    description="API de busca lexical, semantica e hibrida de acordaos do TCU",
    version="2.0.0",
    lifespan=lifespan,
)

# Servir arquivos estáticos (CSS, JS, Imagens)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/")
async def index(background_tasks: BackgroundTasks):
    """Serve a pagina principal e dispara sincronização em background."""
    background_tasks.add_task(_sync_novos_acordaos)
    
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return JSONResponse(
        {"mensagem": "API de Busca de Acordaos do TCU", "docs": "/docs"},
        status_code=200,
    )


class BuscaRequest(BaseModel):
    query: str
    colegiado: str = ""
    tipo_processo: str = ""
    relator: str = ""
    ano: str = ""
    top_k: int = 25
    offset: int = 0
    modo: str = "hibrida"

@app.post("/api/buscar")
def buscar(req: BuscaRequest):
    """Endpoint principal de busca.

    Realiza busca nos acordaos indexados usando o modo especificado.
    """
    conn = _get_conn()

    if req.modo not in ("lexical", "semantica", "hibrida"):
        raise HTTPException(status_code=400, detail="Modo invalido. Use: lexical, semantica ou hibrida")

    try:
        if req.modo == "lexical":
            resultados = busca_lexical(
                conn, req.query, top_k=req.top_k, offset=req.offset,
                ano=req.ano, colegiado=req.colegiado, relator=req.relator, tipo_processo=req.tipo_processo
            )
        elif req.modo == "semantica":
            model = _get_model()
            resultados = busca_semantica(
                conn, req.query, model, top_k=req.top_k, offset=req.offset,
                ano=req.ano, colegiado=req.colegiado, relator=req.relator, tipo_processo=req.tipo_processo
            )
        else:  # hibrida
            model = _get_model()
            resultados = busca_hibrida(
                conn, req.query, model, top_k=req.top_k, offset=req.offset,
                ano=req.ano, colegiado=req.colegiado, relator=req.relator, tipo_processo=req.tipo_processo
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na busca: {str(e)}")

    # Relevancia (1 a 10) derivada do SCORE REAL via normalizacao min-max,
    # e nao da posicao na lista. Preserva as diferencas de qualidade.
    calcular_relevancia(resultados)

    # Mapeamento para compatibilidade com o frontend
    for r in resultados:
        r["key"] = r.get("chave", "")
        r["resumo"] = r.get("trecho", r.get("sumario", ""))

    return {
        "query": req.query,
        "modo": req.modo,
        "top_k": req.top_k,
        "total_candidatos": len(resultados), # O frontend espera total_candidatos
        "resultados": resultados,
    }



# Prazo total (wall-clock) para uma chamada de resumo inteira, do inicio da
# conexao ate o ultimo chunk do streaming. Configuravel via LLM_TIMEOUT_TOTAL
# porque provedores locais (Ollama) e nuvem variam muito em velocidade —
# Ollama tende a ser mais rapido apos o modelo ja estar carregado em memoria,
# mas a primeira chamada pode incluir o tempo de carregar o modelo.
RESUMO_TIMEOUT_TOTAL = float(os.environ.get("LLM_TIMEOUT_TOTAL", "110"))

SYSTEM_PROMPT_RESUMO = (
    "Você é um especialista em jurisprudência do TCU aplicada a licitações e contratos (Lei 14.133/2021).\n\n"
    "REGRAS DE FIDELIDADE E FORMATO (obrigatórias):\n"
    "- Escreva um resumo humanizado em texto corrido (apenas parágrafos coesos). NÃO utilize NENHUMA formatação markdown (como negrito, itálico, listas, subtítulos ou tópicos numerados).\n"
    "- NÃO inclua e não mencione número do acórdão, órgão, colegiado ou relator (pois essa informação já é mostrada no sistema).\n"
    "- Use SOMENTE as informações contidas no texto fornecido abaixo. NÃO utilize conhecimento prévio sobre este ou outros acórdãos.\n"
    "- Artigos de lei, valores monetários e datas devem ser reproduzidos EXATAMENTE como aparecem no texto.\n"
    "- Separe claramente no texto o que é FATO RELATADO do que é sua INTERPRETAÇÃO/aplicação prática.\n\n"
    "O seu resumo deve abordar o caso fluindo organicamente pelos pontos essenciais: "
    "a tese/enunciado principal (regra fixada), o contexto fático do caso, a fundamentação legal citada, "
    "a aplicação prática na rotina de um pregoeiro, e a relevância da decisão (alta, média ou baixa, com justificativa)."
)


def _chamar_llm_streaming(client: OpenAI, modelo_llm: str, conteudo: str, stream_holder: dict) -> str:
    """Executa a chamada de streaming e concatena o texto final.

    Roda em uma thread separada (ver `resumir`) para que o prazo total possa
    ser imposto de fora via `future.result(timeout=...)`, ja que nao ha como
    interromper de forma limpa um `for chunk in stream` preso esperando dados
    que nunca chegam. `stream_holder` recebe o objeto do stream assim que ele
    e criado, para que o chamador possa tentar fecha-lo (`stream.close()`) e
    liberar a conexao mais rapido caso o timeout estoure antes do fim.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_RESUMO},
        {"role": "user", "content": f"Acórdão: {conteudo[:12000]}"},
    ]

    stream = client.chat.completions.create(
        model=modelo_llm,
        messages=messages,
        stream=True,
    )
    stream_holder["stream"] = stream

    content_parts = []
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                content_parts.append(delta.content)

    return "".join(content_parts).strip()


class ResumoRequest(BaseModel):
    chave: str

@app.post("/api/resumir")
def resumir(req: ResumoRequest):
    """Gera um resumo do acórdão usando LLM.

    Provedor, modelo e base_url sao configuraveis por variaveis de ambiente
    (funciona com qualquer backend compativel com a API da OpenAI, incluindo
    Ollama local — ver README para o passo a passo):
        LLM_API_KEY      (fallback: OPENAI_API_KEY; para Ollama, qualquer valor nao vazio serve)
        LLM_BASE_URL     (default: https://api.tokenrouter.com/v1; Ollama: http://localhost:11434/v1)
        LLM_MODEL        (default: z-ai/glm-5.3-free; Ollama: nome do modelo baixado, ex. llama3.1)
        LLM_TIMEOUT_TOTAL (default: 110, em segundos)
    """
    # Aceita LLM_API_KEY (novo) ou OPENAI_API_KEY (compat. legado)
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    placeholders = {"", "sua_chave_aqui", "your_api_key_here", "sk-sua_chave_aqui"}
    if not api_key or api_key.strip() in placeholders:
        raise HTTPException(
            status_code=401,
            detail="Chave de API não configurada. Defina LLM_API_KEY (ou OPENAI_API_KEY) no arquivo .env.",
        )

    base_url = os.environ.get("LLM_BASE_URL", "https://api.tokenrouter.com/v1")
    modelo_llm = os.environ.get("LLM_MODEL", "z-ai/glm-5.3-free")

    conn = _get_conn()
    row = conn.execute("SELECT conteudo FROM acordaos WHERE chave = ?", (req.chave,)).fetchone()

    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Acórdão não encontrado ou sem conteúdo.")

    conteudo = row[0]

    # timeout do cliente: limita cada operacao de rede individual (conexao e o
    # intervalo entre um chunk de streaming e o proximo). Sozinho ele NAO basta
    # como prazo total — um provedor que vai enviando poucos tokens de forma
    # continua nunca fica tempo suficiente em silencio para disparar esse
    # limite, mesmo que o total passe de minutos (foi o que aconteceu com o
    # modelo gratuito padrao). Por isso a chamada roda em outra thread e o
    # prazo real e imposto pelo future.result(timeout=...) abaixo.
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)

    stream_holder: dict = {}
    future = _llm_executor.submit(_chamar_llm_streaming, client, modelo_llm, conteudo, stream_holder)

    try:
        full_content = future.result(timeout=RESUMO_TIMEOUT_TOTAL)
    except FutureTimeoutError:
        # Tenta fechar o stream para liberar a conexao mais cedo; a thread em
        # si continua rodando em segundo plano ate o timeout do cliente HTTP
        # (30s) encerra-la — Python nao permite matar uma thread de fora.
        stream = stream_holder.get("stream")
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        raise HTTPException(
            status_code=504,
            detail=(
                "A IA demorou demais para responder (limite de "
                f"{int(RESUMO_TIMEOUT_TOTAL)}s excedido). Se estiver usando um modelo "
                "local (Ollama), confirme que ele esta rodando e que o modelo ja foi "
                "baixado (`ollama pull <modelo>`); se for um provedor em nuvem, ele "
                "pode estar sobrecarregado — tente novamente em instantes."
            ),
        )
    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="A IA demorou demais para responder. O modelo configurado pode estar sobrecarregado — tente novamente em instantes.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao conectar com a IA: {str(e)}")

    if not full_content:
        raise HTTPException(
            status_code=502,
            detail="A IA retornou uma resposta vazia. Tente novamente.",
        )
    return {"resumo": full_content}

@app.get("/api/filtros")
def filtros():
    """Retorna opcoes de filtros disponiveis para a interface."""
    conn = _get_conn()

    colegiados = [r[0] for r in conn.execute(
        "SELECT DISTINCT colegiado FROM acordaos WHERE colegiado != '' ORDER BY colegiado"
    ).fetchall()]

    tipos_processo = [r[0] for r in conn.execute(
        "SELECT DISTINCT tipo_processo FROM acordaos WHERE tipo_processo != '' ORDER BY tipo_processo"
    ).fetchall()]

    relatores = [r[0] for r in conn.execute(
        "SELECT DISTINCT relator FROM acordaos WHERE relator != '' ORDER BY relator"
    ).fetchall()]

    anos = [r[0] for r in conn.execute(
        "SELECT DISTINCT ano FROM acordaos WHERE ano != '' ORDER BY ano DESC"
    ).fetchall()]

    # Definir macro-temas e as queries FTS (expressões) para buscar no conteúdo
    macro_temas = {
        "Licitação e Contratos": "licita* OR contrato* OR pregão OR certame OR edital",
        "Aposentadoria e Pensão": "aposentadoria OR pensão OR previdenciário OR inativo",
        "Tomada de Contas": "tomada de contas especial OR TCE",
        "Obras Públicas": "obras OR engenharia OR rodovia OR pavimentação",
        "Convênios e Repasses": "convênio OR repasse OR prestação de contas",
        "Fraude e Sobrepreço": "fraude OR sobrepreço OR superfaturamento OR desvio",
        "Pessoal e Concurso": "pessoal OR concurso OR admissão OR remuneração",
        "Auditoria e Inspeção": "auditoria OR inspeção OR fiscalização"
    }

    tema_counts = {}
    for tema, query_fts in macro_temas.items():
        try:
            # Conta quantos acórdãos têm essas palavras no teor
            count = conn.execute("SELECT COUNT(*) FROM acordaos_fts WHERE acordaos_fts MATCH ?", (query_fts,)).fetchone()[0]
            tema_counts[tema] = count
        except Exception:
            tema_counts[tema] = 0

    # Pega os 5 temas com maiores contagens (que não sejam 0)
    top_5_temas = sorted(tema_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_assuntos = [t[0] for t in top_5_temas if t[1] > 0]

    total_acordaos = contar_acordaos(conn)

    return {
        "colegiados": colegiados,
        "tipos_processo": tipos_processo,
        "relatores": relatores,
        "anos": anos,
        "top_assuntos": top_assuntos,
        "total_acordaos": total_acordaos,
    }


@app.get("/api/stats")
def stats():
    """Retorna estatisticas do banco de dados."""
    conn = _get_conn()

    total = contar_acordaos(conn)
    com_embedding = conn.execute(
        "SELECT COUNT(*) FROM acordaos WHERE embedding IS NOT NULL"
    ).fetchone()[0]

    return {
        "total_acordaos": total,
        "com_embedding": com_embedding,
        "sem_embedding": total - com_embedding,
        "percentual_embeddings": round((com_embedding / total * 100) if total > 0 else 0, 1),
    }


@app.get("/api/acordao/{chave}")
def acordao_detalhe(chave: str):
    """Retorna o acordao completo por chave.

    Args:
        chave: Chave unica do acordao (ex: ACORDAO-COMPLETO-2778170).

    Returns:
        JSON com todos os campos do acordao.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, chave, tipo, titulo, numero_acordao, ano, colegiado, "
        "relator, tipo_processo, entidade, assunto, sumario, conteudo "
        "FROM acordaos WHERE chave = ?",
        (chave,),
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Acordao nao encontrado")

    result = dict(row)
    result["decisao"] = result.get("conteudo", "")
    result["numero"] = result.get("numero_acordao", "")
    result["key"] = result.get("chave", "")

    return result

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Servidor iniciando em http://localhost:8000")
    print("[INFO] Documentacao interativa: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
