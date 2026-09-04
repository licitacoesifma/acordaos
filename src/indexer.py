"""
Indexer de Acordaos do TCU.

Cria e gerencia o banco de dados SQLite com tabela principal e indice
FTS5 para busca lexical (BM25).
"""

import os
import sqlite3
from typing import Optional


# Caminho padrao do banco de dados
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db")
DB_PATH = os.path.join(DB_DIR, "acordaos.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Retorna uma conexao com o banco SQLite.

    Args:
        db_path: Caminho para o arquivo do banco de dados.

    Returns:
        Conexao SQLite configurada.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def criar_tabelas(conn: sqlite3.Connection) -> None:
    """Cria a tabela principal e a tabela virtual FTS5.

    Args:
        conn: Conexao SQLite ativa.
    """
    # Tabela principal
    conn.execute("""
        CREATE TABLE IF NOT EXISTS acordaos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chave TEXT UNIQUE NOT NULL,
            tipo TEXT,
            titulo TEXT,
            numero_acordao TEXT,
            ano TEXT,
            colegiado TEXT,
            relator TEXT,
            tipo_processo TEXT,
            entidade TEXT,
            assunto TEXT,
            sumario TEXT,
            conteudo TEXT,
            embedding BLOB
        )
    """)

    # Tabela virtual FTS5 para busca lexical (BM25)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS acordaos_fts USING fts5(
            titulo,
            conteudo,
            entidade,
            relator,
            tipo_processo,
            content='acordaos',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        )
    """)

    # Triggers para manter o FTS sincronizado com a tabela principal
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS acordaos_ai AFTER INSERT ON acordaos BEGIN
            INSERT INTO acordaos_fts(rowid, titulo, conteudo, entidade, relator, tipo_processo)
            VALUES (new.id, new.titulo, new.conteudo, new.entidade, new.relator, new.tipo_processo);
        END;

        CREATE TRIGGER IF NOT EXISTS acordaos_ad AFTER DELETE ON acordaos BEGIN
            INSERT INTO acordaos_fts(acordaos_fts, rowid, titulo, conteudo, entidade, relator, tipo_processo)
            VALUES ('delete', old.id, old.titulo, old.conteudo, old.entidade, old.relator, old.tipo_processo);
        END;

        CREATE TRIGGER IF NOT EXISTS acordaos_au AFTER UPDATE ON acordaos BEGIN
            INSERT INTO acordaos_fts(acordaos_fts, rowid, titulo, conteudo, entidade, relator, tipo_processo)
            VALUES ('delete', old.id, old.titulo, old.conteudo, old.entidade, old.relator, old.tipo_processo);
            INSERT INTO acordaos_fts(rowid, titulo, conteudo, entidade, relator, tipo_processo)
            VALUES (new.id, new.titulo, new.conteudo, new.entidade, new.relator, new.tipo_processo);
        END;
    """)

    # Indices uteis para filtragem
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ano ON acordaos(ano)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relator ON acordaos(relator)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tipo_processo ON acordaos(tipo_processo)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entidade ON acordaos(entidade)")

    conn.commit()
    print("[INFO] Tabelas e indices criados com sucesso.")


def inserir_acordaos(conn: sqlite3.Connection, acordaos: list[dict[str, str]]) -> int:
    """Insere uma lista de acordaos no banco de dados.

    Usa INSERT OR IGNORE para evitar duplicatas pela chave.

    Args:
        conn: Conexao SQLite ativa.
        acordaos: Lista de dicionarios com os campos do acordao.

    Returns:
        Numero de registros efetivamente inseridos.
    """
    sql = """
        INSERT OR IGNORE INTO acordaos
            (chave, tipo, titulo, numero_acordao, ano, colegiado,
             relator, tipo_processo, entidade, assunto, sumario, conteudo)
        VALUES
            (:chave, :tipo, :titulo, :numero_acordao, :ano, :colegiado,
             :relator, :tipo_processo, :entidade, :assunto, :sumario, :conteudo)
    """

    cursor = conn.executemany(sql, acordaos)
    conn.commit()
    inseridos = cursor.rowcount
    print(f"[INFO] {inseridos} acordaos inseridos no banco de dados.")
    return inseridos


def contar_acordaos(conn: sqlite3.Connection) -> int:
    """Retorna o numero total de acordaos no banco.

    Args:
        conn: Conexao SQLite ativa.

    Returns:
        Total de registros na tabela acordaos.
    """
    row = conn.execute("SELECT COUNT(*) FROM acordaos").fetchone()
    return row[0]


def atualizar_embedding(conn: sqlite3.Connection, acordao_id: int, embedding_blob: bytes) -> None:
    """Atualiza o embedding de um acordao especifico.

    Args:
        conn: Conexao SQLite ativa.
        acordao_id: ID do acordao na tabela.
        embedding_blob: Embedding serializado como bytes.
    """
    conn.execute(
        "UPDATE acordaos SET embedding = ? WHERE id = ?",
        (embedding_blob, acordao_id),
    )


def buscar_todos_ids_conteudo(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """Retorna todos os IDs e conteudos para geracao de embeddings.

    Args:
        conn: Conexao SQLite ativa.

    Returns:
        Lista de tuplas (id, conteudo).
    """
    rows = conn.execute("SELECT id, conteudo FROM acordaos ORDER BY id").fetchall()
    return [(row["id"], row["conteudo"]) for row in rows]


# ──────────────────────────────────────────────
# Execucao direta para teste
# ──────────────────────────────────────────────
if __name__ == "__main__":
    conn = get_connection()
    criar_tabelas(conn)
    total = contar_acordaos(conn)
    print(f"[INFO] Total de acordaos no banco: {total}")
    conn.close()
