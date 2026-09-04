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


def carregar_modelo():
    """Carrega o modelo de sentence-transformers.

    Returns:
        Instancia do SentenceTransformer carregada.
    """
    from sentence_transformers import SentenceTransformer

    print(f"[INFO] Carregando modelo: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"[INFO] Modelo carregado. Dimensao do embedding: {model.get_embedding_dimension()}")
    return model


def embedding_para_blob(embedding: np.ndarray) -> bytes:
    """Serializa um vetor numpy float32 para bytes (BLOB).

    Args:
        embedding: Vetor numpy de floats.

    Returns:
        Representacao em bytes do vetor.
    """
    return embedding.astype(np.float32).tobytes()


def blob_para_embedding(blob: bytes) -> np.ndarray:
    """Deserializa bytes (BLOB) para vetor numpy float32.

    Args:
        blob: Bytes armazenados no SQLite.

    Returns:
        Vetor numpy float32 reconstruido.
    """
    return np.frombuffer(blob, dtype=np.float32)


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
    conteudos = [row["conteudo"] for row in rows]

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
