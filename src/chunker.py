"""
Chunker de Acórdãos do TCU.

Lê o arquivo Markdown da pasta data/ e extrai cada acórdão como um
dicionário estruturado. Cada acórdão vira 1 documento (sem quebra interna).
"""

import re
import os
from typing import Optional


def parse_acordaos(filepath: str) -> list[dict[str, str]]:
    """Lê o arquivo Markdown e retorna uma lista de dicionários com os campos de cada acórdão.

    Args:
        filepath: Caminho absoluto para o arquivo .md com os acórdãos.

    Returns:
        Lista de dicionários, cada um representando um acórdão com os campos:
        chave, tipo, titulo, numero_acordao, ano, colegiado, relator,
        tipo_processo, entidade, assunto, sumario, conteudo.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Divide os blocos pelo separador horizontal '---'
    blocos = re.split(r"\n---\n", content)
    acordaos: list[dict[str, str]] = []

    # Mapa de regex para extração dos campos estruturados
    campo_regex: dict[str, str] = {
        "chave":          r"\*\*Chave \(KEY\):\*\*\s*(.+)",
        "tipo":           r"\*\*Tipo:\*\*\s*(.+)",
        "titulo":         r"\*\*Título:\*\*\s*(.+)",
        "numero_acordao": r"\*\*Número do Acórdão:\*\*\s*(\d+)",
        "ano":            r"\*\*Ano:\*\*\s*(\d+)",
        "colegiado":      r"\*\*Colegiado:\*\*\s*(.+)",
        "relator":        r"\*\*Relator:\*\*\s*(.+)",
        "tipo_processo":  r"\*\*Tipo de Processo:\*\*\s*(.+)",
        "entidade":       r"\*\*Entidade:\*\*\s*(.+)",
        "assunto":        r"\*\*Assunto:\*\*\s*(.+)",
        "sumario":        r"\*\*Sumário:\*\*\s*(.+)",
    }

    for bloco in blocos:
        bloco = bloco.strip()

        # Ignora blocos vazios e o cabeçalho do documento
        if not bloco or bloco.startswith("# Jurisprudência"):
            continue

        acordao: dict[str, str] = {}

        # Extrai cada campo usando a regex correspondente
        for campo, pattern in campo_regex.items():
            match = re.search(pattern, bloco)
            acordao[campo] = match.group(1).strip() if match else ""

        # Conteúdo (decisão) — tudo após "**Acórdão (Decisão):**"
        decisao_match = re.search(
            r"\*\*Acórdão \(Decisão\):\*\*\s*\n(.+)", bloco, re.DOTALL
        )
        acordao["conteudo"] = decisao_match.group(1).strip() if decisao_match else ""

        # Só adiciona se a chave for válida (campo obrigatório)
        if acordao.get("chave"):
            acordaos.append(acordao)

    return acordaos


def carregar_todos_acordaos(data_dir: str) -> list[dict[str, str]]:
    """Carrega todos os acórdãos de todos os arquivos .md dentro de data_dir.

    Args:
        data_dir: Caminho para o diretório contendo os arquivos Markdown.

    Returns:
        Lista consolidada de todos os acórdãos encontrados.
    """
    todos: list[dict[str, str]] = []

    if not os.path.exists(data_dir):
        print(f"[ERRO] Diretório não encontrado: {data_dir}")
        return todos

    arquivos_md = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".md")
    )

    if not arquivos_md:
        print(f"[AVISO] Nenhum arquivo .md encontrado em: {data_dir}")
        return todos

    for filename in arquivos_md:
        filepath = os.path.join(data_dir, filename)
        print(f"[INFO] Processando {filename}...")
        acordaos = parse_acordaos(filepath)
        print(f"  -> {len(acordaos)} acordaos extraidos de {filename}")
        todos.extend(acordaos)

    print(f"\n[INFO] Total de acórdãos carregados: {len(todos)}")
    return todos


def imprimir_amostra(acordaos: list[dict[str, str]], n: int = 3) -> None:
    """Imprime os primeiros N acórdãos para validação visual.

    Args:
        acordaos: Lista de acórdãos.
        n: Quantidade de acórdãos a exibir.
    """
    print(f"\n{'='*80}")
    print(f" AMOSTRA: {n} primeiros acórdãos extraídos")
    print(f"{'='*80}")

    for i, ac in enumerate(acordaos[:n]):
        print(f"\n--- Acórdão [{i+1}] ---")
        for campo, valor in ac.items():
            if campo == "conteudo":
                # Mostra apenas os primeiros 200 caracteres do conteúdo
                preview = valor[:200] + "..." if len(valor) > 200 else valor
                print(f"  {campo:18s}: {preview}")
            else:
                print(f"  {campo:18s}: {valor}")
        print()


# ──────────────────────────────────────────────
# Execução direta para validação
# ──────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    print(f"[INFO] Diretório de dados: {DATA_DIR}")
    acordaos = carregar_todos_acordaos(DATA_DIR)

    if acordaos:
        imprimir_amostra(acordaos, n=3)

        # Estatísticas rápidas
        print(f"\n{'='*80}")
        print(f" ESTATÍSTICAS")
        print(f"{'='*80}")
        print(f"  Total de acórdãos: {len(acordaos)}")

        # Campos preenchidos
        campos = ["chave", "tipo", "titulo", "numero_acordao", "ano",
                   "colegiado", "relator", "tipo_processo", "entidade",
                   "assunto", "sumario", "conteudo"]
        for campo in campos:
            preenchidos = sum(1 for ac in acordaos if ac.get(campo))
            pct = (preenchidos / len(acordaos)) * 100
            print(f"  {campo:18s}: {preenchidos:>5d} / {len(acordaos)} ({pct:.1f}%)")

        # Distribuição por colegiado
        colegiados: dict[str, int] = {}
        for ac in acordaos:
            col = ac.get("colegiado", "N/A") or "N/A"
            colegiados[col] = colegiados.get(col, 0) + 1
        print(f"\n  Distribuição por Colegiado:")
        for col, count in sorted(colegiados.items(), key=lambda x: -x[1]):
            print(f"    {col}: {count}")
