"""
Searcher de Acordaos do TCU.

Implementa tres modos de busca:
1. Lexical (FTS5 / BM25)
2. Semantica (similaridade de cosseno com embeddings)
3. Hibrida (Reciprocal Rank Fusion - RRF)
"""

import sqlite3
import threading
from typing import Optional

import numpy as np

from src.indexer import get_connection
from src.embedder import blob_para_embedding, obter_embedding_query, EMBEDDING_DIM


# ──────────────────────────────────────────────
# Configuracoes
# ──────────────────────────────────────────────
RRF_K = 60  # Constante k para Reciprocal Rank Fusion


# ──────────────────────────────────────────────
# Cache da matriz de embeddings (carregada 1x, reusada em todas as buscas)
# ──────────────────────────────────────────────
# Em vez de reler todos os embeddings do SQLite e fazer dot product em loop
# Python a cada requisicao, carregamos uma unica matriz NumPy (N x DIM) e
# calculamos as similaridades de forma vetorizada. O cache e invalidado
# automaticamente quando o numero de acordaos com embedding muda (ex: apos
# uma sincronizacao em background que gerou novos vetores).
_emb_lock = threading.Lock()
_emb_cache: dict = {
    "matriz": None,      # np.ndarray (N, DIM) float32
    "ids": None,         # np.ndarray (N,) int
    "assinatura": None,  # (total_com_embedding,) para deteccao de mudanca
}


def _assinatura_embeddings(conn: sqlite3.Connection) -> int:
    """Retorna a contagem de embeddings, usada para detectar mudancas no cache."""
    return conn.execute(
        "SELECT COUNT(*) FROM acordaos WHERE embedding IS NOT NULL"
    ).fetchone()[0]


def _carregar_matriz_embeddings(conn: sqlite3.Connection):
    """Carrega (ou recarrega) a matriz de embeddings em memoria.

    Thread-safe e idempotente: so recarrega se a assinatura mudou.

    Returns:
        Tupla (matriz NxDIM, ids Nx1) ou (None, None) se nao ha embeddings.
    """
    assinatura = _assinatura_embeddings(conn)

    with _emb_lock:
        if _emb_cache["assinatura"] == assinatura and _emb_cache["matriz"] is not None:
            return _emb_cache["matriz"], _emb_cache["ids"]

        rows = conn.execute(
            "SELECT id, embedding FROM acordaos WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()

        ids_validos = []
        vetores = []
        for row in rows:
            emb = blob_para_embedding(row["embedding"])
            if emb is None:  # pula embeddings corrompidos/dimensao errada
                continue
            ids_validos.append(row["id"])
            vetores.append(emb)

        if not vetores:
            _emb_cache.update(matriz=None, ids=None, assinatura=assinatura)
            return None, None

        matriz = np.vstack(vetores).astype(np.float32)
        ids = np.array(ids_validos, dtype=np.int64)

        _emb_cache.update(matriz=matriz, ids=ids, assinatura=assinatura)
        return matriz, ids


def invalidar_cache_embeddings() -> None:
    """Forca o recarregamento da matriz na proxima busca semantica.

    Deve ser chamada apos gerar novos embeddings (ex: sincronizacao).
    """
    with _emb_lock:
        _emb_cache.update(matriz=None, ids=None, assinatura=None)


def busca_lexical(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 25,
    offset: int = 0,
    ano: str = "",
    colegiado: str = "",
    relator: str = "",
    tipo_processo: str = "",
    incluir_trecho: bool = True,
) -> list[dict]:
    """Busca lexical usando FTS5 com ranking BM25.

    Args:
        conn: Conexao SQLite ativa.
        query: Texto da busca.
        top_k: Numero maximo de resultados.
        incluir_trecho: Se False, nao busca/computa a coluna 'conteudo' nem o
            trecho de destaque. Usado pela busca hibrida, que descarta a
            maioria dos candidatos intermediarios apos a fusao RRF — buscar
            e processar o texto completo deles seria desperdicio.

    Returns:
        Lista de dicionarios com os resultados, incluindo score BM25.
    """
    conteudo_select = "a.conteudo,\n            " if incluir_trecho else ""
    sql = f"""
        SELECT
            a.id,
            a.chave,
            a.titulo,
            a.tipo,
            a.numero_acordao,
            a.ano,
            a.colegiado,
            a.relator,
            a.tipo_processo,
            a.entidade,
            a.assunto,
            a.sumario,
            {conteudo_select}fts.rank AS score
        FROM acordaos_fts fts
        JOIN acordaos a ON a.id = fts.rowid
        WHERE acordaos_fts MATCH ?
          AND (? = '' OR a.ano = ?)
          AND (? = '' OR a.colegiado = ?)
          AND (? = '' OR a.relator = ?)
          AND (? = '' OR a.tipo_processo = ?)
        ORDER BY fts.rank
        LIMIT ? OFFSET ?
    """

    params = (
        query,
        ano, ano,
        colegiado, colegiado,
        relator, relator,
        tipo_processo, tipo_processo,
        top_k, offset
    )

    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        # Fallback: tenta escapar a query para FTS5
        query_escaped = " ".join(
            f'"{termo}"' for termo in query.split() if termo.strip()
        )
        if not query_escaped:
            return []
        try:
            params_escaped = (
                query_escaped,
                ano, ano,
                colegiado, colegiado,
                relator, relator,
                tipo_processo, tipo_processo,
                top_k, offset
            )
            rows = conn.execute(sql, params_escaped).fetchall()
        except Exception:
            return []

    resultados = []
    for row in rows:
        resultado = dict(row)
        # BM25 retorna valores negativos (mais negativo = mais relevante)
        resultado["score"] = -resultado["score"]
        if incluir_trecho:
            resultado["trecho"] = _extrair_trecho(resultado["conteudo"], query)
            resultado.pop("conteudo", None)
        resultados.append(resultado)

    return resultados


def busca_semantica(
    conn: sqlite3.Connection,
    query: str,
    model,
    top_k: int = 25,
    offset: int = 0,
    ano: str = "",
    colegiado: str = "",
    relator: str = "",
    tipo_processo: str = "",
    incluir_trecho: bool = True,
) -> list[dict]:
    """Busca semantica usando similaridade de cosseno com embeddings.

    Args:
        conn: Conexao SQLite ativa.
        query: Texto da busca.
        model: Instancia do SentenceTransformer.
        top_k: Numero maximo de resultados.
        incluir_trecho: Se False, nao busca a coluna 'conteudo' nem computa o
            trecho de destaque (ver busca_lexical).

    Returns:
        Lista de dicionarios com os resultados, incluindo score de similaridade.
    """
    query = (query or "").strip()
    if not query:
        return []

    # Gera embedding da query (normalizado)
    query_embedding = np.asarray(
        obter_embedding_query(model, query), dtype=np.float32
    )

    # Carrega a matriz de embeddings em memoria (cacheada)
    matriz, ids = _carregar_matriz_embeddings(conn)
    if matriz is None or matriz.shape[0] == 0:
        return []

    # Similaridade de cosseno vetorizada: como tudo esta normalizado,
    # cosseno = produto interno. matriz (N, DIM) @ query (DIM,) -> (N,)
    sims = matriz @ query_embedding

    # Se ha filtros, restringe aos IDs permitidos via SQL
    tem_filtro = any([ano, colegiado, relator, tipo_processo])
    ids_permitidos: Optional[set] = None
    if tem_filtro:
        sql_ids = """
            SELECT id FROM acordaos
            WHERE embedding IS NOT NULL
              AND (? = '' OR ano = ?)
              AND (? = '' OR colegiado = ?)
              AND (? = '' OR relator = ?)
              AND (? = '' OR tipo_processo = ?)
        """
        params = (
            ano, ano,
            colegiado, colegiado,
            relator, relator,
            tipo_processo, tipo_processo,
        )
        ids_permitidos = {r[0] for r in conn.execute(sql_ids, params).fetchall()}
        if not ids_permitidos:
            return []

    # Ordena todos os indices por similaridade decrescente (argsort e barato)
    ordem = np.argsort(-sims)

    # Coleta os IDs+scores finais respeitando filtro, offset e top_k
    selecionados: list[tuple[int, float]] = []
    pulados = 0
    for idx in ordem:
        acordao_id = int(ids[idx])
        if ids_permitidos is not None and acordao_id not in ids_permitidos:
            continue
        if pulados < offset:
            pulados += 1
            continue
        selecionados.append((acordao_id, float(sims[idx])))
        if len(selecionados) >= top_k:
            break

    if not selecionados:
        return []

    # Busca os metadados apenas dos selecionados (1 query, preservando a ordem)
    id_lista = [sid for sid, _ in selecionados]
    placeholders = ",".join("?" for _ in id_lista)
    conteudo_col = ", conteudo" if incluir_trecho else ""
    linhas = conn.execute(
        f"""
        SELECT id, chave, titulo, tipo, numero_acordao, ano, colegiado,
               relator, tipo_processo, entidade, assunto, sumario{conteudo_col}
        FROM acordaos WHERE id IN ({placeholders})
        """,
        id_lista,
    ).fetchall()
    por_id = {row["id"]: dict(row) for row in linhas}

    resultados = []
    for acordao_id, score in selecionados:
        dados = por_id.get(acordao_id)
        if dados is None:
            continue
        dados["score"] = score
        if incluir_trecho:
            dados["trecho"] = _extrair_trecho(dados.get("conteudo", ""), query)
            dados.pop("conteudo", None)
        resultados.append(dados)

    return resultados


def busca_hibrida(
    conn: sqlite3.Connection,
    query: str,
    model,
    top_k: int = 25,
    offset: int = 0,
    peso_lexical: float = 0.4,
    peso_semantico: float = 0.6,
    ano: str = "",
    colegiado: str = "",
    relator: str = "",
    tipo_processo: str = "",
) -> list[dict]:
    """Busca hibrida combinando resultados lexicais e semanticos via RRF.

    Usa Reciprocal Rank Fusion (RRF) para combinar os rankings de ambas
    as buscas, com pesos configuraveis.

    Args:
        conn: Conexao SQLite ativa.
        query: Texto da busca.
        model: Instancia do SentenceTransformer.
        top_k: Numero maximo de resultados finais.
        peso_lexical: Peso da busca lexical no score final.
        peso_semantico: Peso da busca semantica no score final.

    Returns:
        Lista de dicionarios com os resultados combinados.
    """
    if not (query or "").strip():
        return []

    # Busca mais resultados intermediarios para ter uma boa cobertura na fusao.
    # Minimo de 50 garante fusao util mesmo para top_k pequeno. incluir_trecho=False
    # evita buscar/processar o texto completo (conteudo + extracao de trecho) para
    # esses candidatos intermediarios: a maioria e descartada apos a fusao RRF, entao
    # calcular o trecho deles seria trabalho jogado fora.
    search_scope = max(50, (offset + top_k) * 3)
    resultados_lex = busca_lexical(
        conn, query, top_k=search_scope, offset=0,
        ano=ano, colegiado=colegiado, relator=relator, tipo_processo=tipo_processo,
        incluir_trecho=False,
    )
    resultados_sem = busca_semantica(
        conn, query, model, top_k=search_scope, offset=0,
        ano=ano, colegiado=colegiado, relator=relator, tipo_processo=tipo_processo,
        incluir_trecho=False,
    )

    # Mapa chave -> dados do acordao (para metadados)
    dados_acordaos: dict[str, dict] = {}

    # Calcula scores RRF
    scores_rrf: dict[str, float] = {}

    # RRF para resultados lexicais
    for rank, resultado in enumerate(resultados_lex, start=1):
        chave = resultado["chave"]
        rrf_score = peso_lexical * (1.0 / (RRF_K + rank))
        scores_rrf[chave] = scores_rrf.get(chave, 0.0) + rrf_score
        if chave not in dados_acordaos:
            dados_acordaos[chave] = resultado

    # RRF para resultados semanticos
    for rank, resultado in enumerate(resultados_sem, start=1):
        chave = resultado["chave"]
        rrf_score = peso_semantico * (1.0 / (RRF_K + rank))
        scores_rrf[chave] = scores_rrf.get(chave, 0.0) + rrf_score
        if chave not in dados_acordaos:
            dados_acordaos[chave] = resultado

    # Ordena pelo score RRF combinado
    ranking = sorted(scores_rrf.items(), key=lambda x: x[1], reverse=True)
    pagina = ranking[offset:offset + top_k]

    # So agora, com o conjunto final ja definido (top_k, nao search_scope*2),
    # busca o conteudo e computa o trecho de destaque para quem realmente vai
    # ser retornado.
    chaves_finais = [chave for chave, _ in pagina]
    conteudo_por_chave: dict[str, str] = {}
    if chaves_finais:
        placeholders = ",".join("?" for _ in chaves_finais)
        linhas = conn.execute(
            f"SELECT chave, conteudo FROM acordaos WHERE chave IN ({placeholders})",
            chaves_finais,
        ).fetchall()
        conteudo_por_chave = {row["chave"]: row["conteudo"] for row in linhas}

    resultados = []
    for chave, score in pagina:
        dados = dados_acordaos[chave].copy()
        dados["score"] = round(score, 6)
        dados["metodo"] = "hibrido"
        dados["trecho"] = _extrair_trecho(conteudo_por_chave.get(chave, ""), query)
        resultados.append(dados)

    return resultados


def calcular_relevancia(resultados: list[dict]) -> None:
    """Adiciona um campo 'relevancia' (1 a 10) baseado no SCORE REAL.

    Aplica normalizacao min-max sobre os scores efetivamente retornados,
    preservando as diferencas relativas de qualidade entre os resultados.
    Modifica a lista in-place. Diferente de uma escala por posicao, aqui
    dois resultados com scores muito proximos recebem relevancias proximas,
    e um resultado fraco no topo NAO recebe nota maxima artificialmente.

    Args:
        resultados: Lista de dicionarios contendo a chave 'score'.
    """
    if not resultados:
        return

    scores = [float(r.get("score", 0.0)) for r in resultados]
    s_min, s_max = min(scores), max(scores)
    intervalo = s_max - s_min

    for r, s in zip(resultados, scores):
        if intervalo <= 1e-12:
            # Todos os scores praticamente iguais: relevancia uniforme alta
            r["relevancia"] = 10
        else:
            norm = (s - s_min) / intervalo  # 0.0 .. 1.0
            r["relevancia"] = int(round(1 + norm * 9))  # 1 .. 10


def _extrair_trecho(conteudo: str, query: str, tamanho: int = 300) -> str:
    """Extrai um trecho relevante do conteudo ao redor do termo buscado.

    Args:
        conteudo: Texto completo do acordao.
        query: Texto da busca.
        tamanho: Tamanho maximo do trecho em caracteres.

    Returns:
        Trecho do texto mais relevante.
    """
    if not conteudo:
        return ""

    conteudo_lower = conteudo.lower()
    termos = query.lower().split()

    # Tenta encontrar o primeiro termo no conteudo
    melhor_pos = -1
    for termo in termos:
        pos = conteudo_lower.find(termo)
        if pos != -1:
            melhor_pos = pos
            break

    if melhor_pos == -1:
        # Se nao encontrou nenhum termo, retorna o inicio
        return conteudo[:tamanho] + ("..." if len(conteudo) > tamanho else "")

    # Centraliza o trecho ao redor do termo encontrado
    inicio = max(0, melhor_pos - tamanho // 3)
    fim = min(len(conteudo), inicio + tamanho)

    trecho = conteudo[inicio:fim]

    # Adiciona reticencias se cortou
    if inicio > 0:
        trecho = "..." + trecho
    if fim < len(conteudo):
        trecho = trecho + "..."

    return trecho


def formatar_resultado(resultado: dict, posicao: int) -> str:
    """Formata um resultado para exibicao no terminal.

    Args:
        resultado: Dicionario com os dados do resultado.
        posicao: Posicao no ranking (1-based).

    Returns:
        String formatada para exibicao.
    """
    linhas = [
        f"\n{'='*70}",
        f"  #{posicao} | Score: {resultado.get('score', 0):.4f}",
        f"{'='*70}",
        f"  Titulo:    {resultado.get('titulo', 'N/A')}",
        f"  Chave:     {resultado.get('chave', 'N/A')}",
        f"  Relator:   {resultado.get('relator', 'N/A')}",
        f"  Processo:  {resultado.get('tipo_processo', 'N/A')}",
        f"  Entidade:  {resultado.get('entidade', 'N/A')}",
        f"  Ano:       {resultado.get('ano', 'N/A')}",
        f"  Colegiado: {resultado.get('colegiado', 'N/A')}",
    ]

    trecho = resultado.get("trecho", "")
    if trecho:
        linhas.append(f"\n  Trecho:")
        # Quebra o trecho em linhas de ~80 caracteres
        palavras = trecho.split()
        linha_atual = "    "
        for palavra in palavras:
            if len(linha_atual) + len(palavra) > 80:
                linhas.append(linha_atual)
                linha_atual = "    " + palavra
            else:
                linha_atual += " " + palavra
        if linha_atual.strip():
            linhas.append(linha_atual)

    return "\n".join(linhas)


# ──────────────────────────────────────────────
# Execucao direta para teste
# ──────────────────────────────────────────────
if __name__ == "__main__":
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM acordaos").fetchone()[0]
    com_emb = conn.execute(
        "SELECT COUNT(*) FROM acordaos WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    print(f"[INFO] Acordaos no banco: {total} | Com embedding: {com_emb}")

    if total > 0:
        query = "licitacao irregularidade"
        print(f"\n[TESTE] Busca lexical: '{query}'")
        resultados = busca_lexical(conn, query, top_k=3)
        for i, r in enumerate(resultados, 1):
            print(formatar_resultado(r, i))

    conn.close()
