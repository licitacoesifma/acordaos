"""
Searcher de Acordaos do TCU.

Implementa tres modos de busca:
1. Lexical (FTS5 / BM25)
2. Semantica (similaridade de cosseno com embeddings)
3. Hibrida (Reciprocal Rank Fusion - RRF)
"""

import sqlite3
from typing import Optional

import numpy as np

from src.indexer import get_connection
from src.embedder import blob_para_embedding, obter_embedding_query


# ──────────────────────────────────────────────
# Configuracoes
# ──────────────────────────────────────────────
RRF_K = 60  # Constante k para Reciprocal Rank Fusion


def busca_lexical(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 25,
    offset: int = 0,
    ano: str = "",
    colegiado: str = "",
    relator: str = "",
    tipo_processo: str = "",
) -> list[dict]:
    """Busca lexical usando FTS5 com ranking BM25.

    Args:
        conn: Conexao SQLite ativa.
        query: Texto da busca.
        top_k: Numero maximo de resultados.

    Returns:
        Lista de dicionarios com os resultados, incluindo score BM25.
    """
    sql = """
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
            a.conteudo,
            fts.rank AS score
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
) -> list[dict]:
    """Busca semantica usando similaridade de cosseno com embeddings.

    Args:
        conn: Conexao SQLite ativa.
        query: Texto da busca.
        model: Instancia do SentenceTransformer.
        top_k: Numero maximo de resultados.

    Returns:
        Lista de dicionarios com os resultados, incluindo score de similaridade.
    """
    # Gera embedding da query
    query_embedding = obter_embedding_query(model, query)

    # Busca embeddings filtrados
    sql = """
        SELECT id, chave, titulo, tipo, numero_acordao, ano, colegiado,
               relator, tipo_processo, entidade, assunto, sumario, conteudo, embedding
        FROM acordaos
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
        tipo_processo, tipo_processo
    )
    
    rows = conn.execute(sql, params).fetchall()

    if not rows:
        return []

    # Calcula similaridade de cosseno para cada acordao
    scores = []
    for row in rows:
        emb = blob_para_embedding(row["embedding"])
        # Como os embeddings ja sao normalizados, cosseno = dot product
        sim = float(np.dot(query_embedding, emb))
        scores.append((sim, dict(row)))

    # Ordena por similaridade decrescente
    scores.sort(key=lambda x: x[0], reverse=True)

    resultados = []
    for score, dados in scores[offset:offset + top_k]:
        dados["score"] = score
        dados["trecho"] = _extrair_trecho(dados["conteudo"], query)
        dados.pop("conteudo", None)
        dados.pop("embedding", None)
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
    # Busca mais resultados intermediarios para ter uma boa cobertura
    search_scope = (offset + top_k) * 3
    resultados_lex = busca_lexical(
        conn, query, top_k=search_scope, offset=0,
        ano=ano, colegiado=colegiado, relator=relator, tipo_processo=tipo_processo
    )
    resultados_sem = busca_semantica(
        conn, query, model, top_k=search_scope, offset=0,
        ano=ano, colegiado=colegiado, relator=relator, tipo_processo=tipo_processo
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

    resultados = []
    for chave, score in ranking[offset:offset + top_k]:
        dados = dados_acordaos[chave].copy()
        dados["score"] = round(score, 6)
        dados["metodo"] = "hibrido"
        resultados.append(dados)

    return resultados


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
