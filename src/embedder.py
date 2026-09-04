"""
Embedder de Acordaos do TCU.

Gera embeddings vetoriais usando o modelo
SamuelMauli/parity-embedding-juridico-br-v4
e armazena como BLOB na tabela acordaos.
"""

import os
import struct
import sqlite3
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.indexer import get_connection, atualizar_embedding, buscar_todos_ids_conteudo


# ──────────────────────────────────────────────
# Configuracao do modelo
# ──────────────────────────────────────────────
MODEL_NAME = "SamuelMauli/parity-embedding-juridico-br-v4"
BATCH_SIZE = 32  # Tamanho do batch para encoding
EMBEDDING_DIM = 384  # Dimensao esperada do vetor (compativel com vector(384) do pgvector)


def carregar_modelo():
    """Carrega o modelo de sentence-transformers.

    Returns:
        Instancia do SentenceTransformer carregada.
    """
    from sentence_transformers import SentenceTransformer

    print(f"[INFO] Carregando modelo: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # get_sentence_embedding_dimension() e a API correta do sentence-transformers.
    # Protegido para nunca derrubar o carregamento por causa de um log informativo.
    try:
        dim = model.get_sentence_embedding_dimension()
        print(f"[INFO] Modelo carregado. Dimensao do embedding: {dim}")
    except Exception:
        print("[INFO] Modelo carregado.")

    return model


def embedding_para_blob(embedding: np.ndarray) -> bytes:
    """Serializa um vetor numpy float32 para bytes (BLOB).

    Args:
        embedding: Vetor numpy de floats.

    Returns:
        Representacao em bytes do vetor (contiguo, float32).
    """
    # np.ascontiguousarray garante buffer contiguo antes de .tobytes()
    return np.ascontiguousarray(embedding, dtype=np.float32).tobytes()


def blob_para_embedding(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """Deserializa bytes (BLOB) para vetor numpy float32.

    Retorna None quando o blob esta ausente ou corrompido (tamanho
    incompativel com a dimensao esperada), permitindo que o chamador
    pule silenciosamente registros invalidos em vez de quebrar a busca.

    Args:
        blob: Bytes armazenados no SQLite (ou None).

    Returns:
        Vetor numpy float32 reconstruido, ou None se invalido.
    """
    if not blob:
        return None

    # np.frombuffer exige que len(blob) seja multiplo de 4 (float32);
    # bytes corrompidos disparam ValueError, entao protegemos.
    try:
        vetor = np.frombuffer(blob, dtype=np.float32)
    except ValueError:
        return None

    # Valida o tamanho contra a dimensao esperada.
    if vetor.shape[0] != EMBEDDING_DIM:
        return None

    return vetor


def gerar_embeddings(conn: sqlite3.Connection, model=None) -> int:
    """Gera embeddings para todos os acordaos que ainda nao possuem.

    Args:
        conn: Conexao SQLite ativa.
        model: Instancia do SentenceTransformer (carrega automaticamente se None).

    Returns:
        Numero de embeddings gerados.
    """
    if model is None:
        model = carregar_modelo()

    # Busca apenas acordaos sem embedding
    rows = conn.execute(
        "SELECT id, conteudo FROM acordaos WHERE embedding IS NULL ORDER BY id"
    ).fetchall()

    if not rows:
        print("[INFO] Todos os acordaos ja possuem embeddings.")
        return 0

    ids = [row["id"] for row in rows]
    # Conteudo vazio geraria um vetor degenerado; usa o titulo como fallback
    # so no encode (nao altera o que esta salvo na coluna conteudo).
    conteudos = [(row["conteudo"] or "").strip() or "documento sem conteudo" for row in rows]

    print(f"[INFO] Gerando embeddings para {len(conteudos)} acordaos...")

    # Gera embeddings em batches com barra de progresso
    total_gerados = 0
    for i in tqdm(range(0, len(conteudos), BATCH_SIZE), desc="Embeddings", unit="batch"):
        batch_textos = conteudos[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]

        embeddings = model.encode(
            batch_textos,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for acordao_id, emb in zip(batch_ids, embeddings):
            blob = embedding_para_blob(emb)
            atualizar_embedding(conn, acordao_id, blob)

        # Commit por batch: se o processo cair no meio de uma base grande,
        # o progresso ja gravado nao e perdido (retomavel na proxima execucao).
        conn.commit()
        total_gerados += len(batch_ids)

    conn.commit()
    print(f"[INFO] {total_gerados} embeddings gerados e salvos.")
    return total_gerados


def obter_embedding_query(model, query: str) -> np.ndarray:
    """Gera o embedding de uma query de busca.

    Args:
        model: Instancia do SentenceTransformer.
        query: Texto da busca.

    Returns:
        Vetor numpy normalizado.
    """
    embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding


# ──────────────────────────────────────────────
# Execucao direta para teste
# ──────────────────────────────────────────────
if __name__ == "__main__":
    conn = get_connection()

    # Verifica quantos ja tem embedding
    total = conn.execute("SELECT COUNT(*) FROM acordaos").fetchone()[0]
    com_emb = conn.execute("SELECT COUNT(*) FROM acordaos WHERE embedding IS NOT NULL").fetchone()[0]
    print(f"[INFO] Acordaos no banco: {total} | Com embedding: {com_emb}")

    if total > 0 and com_emb < total:
        model = carregar_modelo()
        gerar_embeddings(conn, model)
    elif total == 0:
        print("[AVISO] Nenhum acordao no banco. Execute a indexacao primeiro.")

    conn.close()
