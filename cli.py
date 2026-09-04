"""
Script principal do Sistema de Busca de Acordaos do TCU.

Uso:
    python cli.py --indexar              Chunking + indexacao SQLite + embeddings
    python cli.py --buscar "termo"       Busca hibrida nos acordaos
    python cli.py --buscar "termo" -k 5  Define o numero de resultados
    python cli.py --buscar "termo" --modo lexical   Apenas busca lexical
    python cli.py --buscar "termo" --modo semantica Apenas busca semantica
"""

import argparse
import os
import sys
import time


def executar_indexacao() -> None:
    """Executa o pipeline completo de indexacao: chunking + SQLite + embeddings."""
    from src.chunker import carregar_todos_acordaos, imprimir_amostra
    from src.indexer import get_connection, criar_tabelas, inserir_acordaos, contar_acordaos
    from src.embedder import gerar_embeddings, carregar_modelo

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    print("=" * 70)
    print("  ETAPA 1/3: Chunking do arquivo Markdown")
    print("=" * 70)
    inicio = time.time()
    acordaos = carregar_todos_acordaos(DATA_DIR)

    if not acordaos:
        print("[ERRO] Nenhum acordao encontrado. Verifique a pasta data/")
        sys.exit(1)

    imprimir_amostra(acordaos, n=2)
    print(f"  Tempo: {time.time() - inicio:.1f}s")

    print("\n" + "=" * 70)
    print("  ETAPA 2/3: Indexacao no SQLite + FTS5")
    print("=" * 70)
    inicio = time.time()
    conn = get_connection()
    criar_tabelas(conn)
    inserir_acordaos(conn, acordaos)
    total = contar_acordaos(conn)
    print(f"[INFO] Total no banco apos indexacao: {total}")
    print(f"  Tempo: {time.time() - inicio:.1f}s")

    print("\n" + "=" * 70)
    print("  ETAPA 3/3: Geracao de Embeddings")
    print("=" * 70)
    inicio = time.time()
    model = carregar_modelo()
    gerados = gerar_embeddings(conn, model)
    print(f"  Tempo: {time.time() - inicio:.1f}s")

    conn.close()
    print("\n" + "=" * 70)
    print("  INDEXACAO CONCLUIDA COM SUCESSO!")
    print("=" * 70)


def executar_busca(query: str, top_k: int = 10, modo: str = "hibrida") -> None:
    """Executa uma busca nos acordaos indexados.

    Args:
        query: Texto da busca.
        top_k: Numero maximo de resultados.
        modo: Tipo de busca ('lexical', 'semantica', 'hibrida').
    """
    from src.indexer import get_connection
    from src.searcher import busca_lexical, busca_semantica, busca_hibrida, formatar_resultado

    conn = get_connection()

    total = conn.execute("SELECT COUNT(*) FROM acordaos").fetchone()[0]
    if total == 0:
        print("[ERRO] Banco de dados vazio. Execute 'python cli.py --indexar' primeiro.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  Busca: '{query}' | Modo: {modo} | Top-K: {top_k}")
    print(f"{'='*70}")

    inicio = time.time()

    if modo == "lexical":
        resultados = busca_lexical(conn, query, top_k=top_k)
    elif modo == "semantica":
        from src.embedder import carregar_modelo
        model = carregar_modelo()
        resultados = busca_semantica(conn, query, model, top_k=top_k)
    else:  # hibrida
        from src.embedder import carregar_modelo
        model = carregar_modelo()
        resultados = busca_hibrida(conn, query, model, top_k=top_k)

    tempo = time.time() - inicio

    if not resultados:
        print("\n  Nenhum resultado encontrado.")
    else:
        for i, resultado in enumerate(resultados, 1):
            print(formatar_resultado(resultado, i))

    print(f"\n{'='*70}")
    print(f"  {len(resultados)} resultados em {tempo:.2f}s")
    print(f"{'='*70}")

    conn.close()


def main() -> None:
    """Ponto de entrada principal com parsing de argumentos."""
    parser = argparse.ArgumentParser(
        description="Sistema de Busca de Acordaos do TCU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python cli.py --indexar
  python cli.py --buscar "licitacao irregularidade"
  python cli.py --buscar "dano ao erario" -k 5
  python cli.py --buscar "pregao eletronico" --modo lexical
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--indexar",
        action="store_true",
        help="Executa o pipeline completo de indexacao",
    )
    group.add_argument(
        "--buscar",
        type=str,
        metavar="QUERY",
        help="Realiza busca nos acordaos indexados",
    )

    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="Numero maximo de resultados (padrao: 10)",
    )
    parser.add_argument(
        "--modo",
        type=str,
        choices=["lexical", "semantica", "hibrida"],
        default="hibrida",
        help="Modo de busca (padrao: hibrida)",
    )

    args = parser.parse_args()

    if args.indexar:
        executar_indexacao()
    elif args.buscar:
        executar_busca(args.buscar, top_k=args.top_k, modo=args.modo)


if __name__ == "__main__":
    main()
