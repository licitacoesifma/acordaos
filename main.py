"""
API FastAPI para Busca de Acordaos do TCU.

Endpoints:
    GET  /              -> Serve o index.html
    GET  /buscar        -> Busca hibrida nos acordaos
    GET  /api/filtros   -> Retorna opcoes de filtros disponiveis
    GET  /api/stats     -> Estatisticas do banco de dados

Uso:
    python server.py
    # ou
    uvicorn server:app --reload --port 5000
"""

import os
import sys
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
from openai import OpenAI

from src.indexer import get_connection, contar_acordaos, inserir_acordaos
from src.searcher import busca_lexical, busca_semantica, busca_hibrida
from src.chunker import carregar_todos_acordaos
from src.embedder import gerar_embeddings

# Carrega variáveis de ambiente
load_dotenv()

# ──────────────────────────────────────────────
# Estado global da aplicacao
# ──────────────────────────────────────────────
_model = None
_conn = None
_sync_in_progress = False

def _get_model():
    """Carrega o modelo de embeddings (lazy loading)."""
    global _model
    if _model is None:
        from src.embedder import carregar_modelo
        _model = carregar_modelo()
    return _model

def _get_conn():
    """Retorna a conexao com o banco de dados."""
    global _conn
    if _conn is None:
        _conn = get_connection()
    return _conn

def _sync_novos_acordaos():
    """Tarefa em background para buscar novos arquivos em data/ e indexar."""
    global _sync_in_progress
    if _sync_in_progress:
        return
        
    _sync_in_progress = True
    try:
        conn = _get_conn()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, "data")
        
        # 1. Carrega todos os acórdãos dos arquivos MD
        acordaos = carregar_todos_acordaos(DATA_DIR)
        if not acordaos:
            return
            
        # 2. Insere no banco (INSERT OR IGNORE cuida para não duplicar)
        inseridos = inserir_acordaos(conn, acordaos)
        
        # 3. Se houve novos inseridos ou há acórdãos sem embedding, atualiza
        com_embedding = conn.execute("SELECT COUNT(*) FROM acordaos WHERE embedding IS NOT NULL").fetchone()[0]
        total = contar_acordaos(conn)
        
        if inseridos > 0 or com_embedding < total:
            model = _get_model()
            gerar_embeddings(conn, model)
            print(f"[INFO] Sincronização concluída. {total - com_embedding} novos embeddings gerados.")
            
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
        print("[AVISO] Banco vazio. Execute 'python main.py --indexar' primeiro.")

    yield

    # Shutdown
    global _conn
    if _conn:
        _conn.close()
        _conn = None
    print("[INFO] Conexao com banco de dados encerrada.")


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
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return JSONResponse(
        {"mensagem": "API de Busca de Acordaos do TCU", "docs": "/docs"},
        status_code=200,
    )


from pydantic import BaseModel

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
async def buscar(req: BuscaRequest):
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

    # Mapeamento para compatibilidade com o frontend antigo
    for i, r in enumerate(resultados):
        r["key"] = r.get("chave", "")
        r["resumo"] = r.get("trecho", r.get("sumario", ""))
        
        # Mapeia a relevancia para uma escala de 1 a 10 esperada pelo frontend
        # Para evitar complexidade de normalizar scores lexicais/semanticos,
        # geramos uma relevancia baseada na posicao global (offset + i)
        r["relevancia"] = max(1, 10 - (req.offset + i))

    return {
        "query": req.query,
        "modo": req.modo,
        "top_k": req.top_k,
        "total_candidatos": len(resultados), # O frontend espera total_candidatos
        "resultados": resultados,
    }



class ResumoRequest(BaseModel):
    chave: str

@app.post("/api/resumir")
async def resumir(req: ResumoRequest):
    """Gera um resumo do acórdão usando LLM."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "sua_chave_aqui":
        raise HTTPException(status_code=401, detail="API Key não configurada. Defina OPENAI_API_KEY no arquivo .env.")

    conn = _get_conn()
    row = conn.execute("SELECT conteudo FROM acordaos WHERE chave = ?", (req.chave,)).fetchone()
    
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Acórdão não encontrado ou sem conteúdo.")

    conteudo = row[0]

    try:
        client = OpenAI(
            base_url='https://api.tokenrouter.com/v1',
            api_key=api_key,
        )

        messages = [
            {
                "role": "system", 
                "content": (
                    "Você é um especialista jurídico do Tribunal de Contas da União. "
                    "Faça um resumo DIRETO, SÓLIDO e MUITO CONCISO (máximo 3 parágrafos fluidos) focado "
                    "exclusivamente em responder:\n"
                    "1. Do que se trata o acórdão (o fato principal).\n"
                    "2. Quais as consequências geradas (impactos/irregularidades).\n"
                    "3. O Veredito final (o que foi aplicado aos responsáveis e o porquê).\n"
                    "Não use Markdown, listas ou bullet points. Escreva apenas os parágrafos corridos respondendo a essas questões de forma madura."
                )
            },
            {"role": "user", "content": f"Acórdão: {conteudo[:12000]}"},
        ]

        stream = client.chat.completions.create(
            model="z-ai/glm-5.3-free",
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={}
        )

        content_parts = []
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content_parts.append(delta.content)

        full_content = "".join(content_parts)
        return {"resumo": full_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao conectar com a IA: {str(e)}")

@app.get("/api/filtros")
async def filtros():
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

    return {
        "colegiados": colegiados,
        "tipos_processo": tipos_processo,
        "relatores": relatores,
        "anos": anos,
        "top_assuntos": top_assuntos,
        "total_acordaos": contar_acordaos(conn),
    }


@app.get("/api/stats")
async def stats():
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
async def acordao_detalhe(chave: str):
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
    
    return result

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Servidor iniciando em http://localhost:5000")
    print("[INFO] Documentacao interativa: http://localhost:5000/docs")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
