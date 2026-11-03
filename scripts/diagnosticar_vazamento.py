#!/usr/bin/env python3
"""
Diagnostico do vazamento de ementa: o que a Etapa 2 esta marcando, e por que.

Responde a tres perguntas, com numeros e exemplos:
1. A ementa aparece INTEIRA como um bloco dentro do inteiro teor? Quantas vezes e onde
   (inicio, meio ou fim do documento)? Isso e o "chunk 0" classico: copia formal.
2. Ou so FRASES da ementa aparecem no teor, espalhadas? Isso indica ementa redigida
   copiando trechos do voto: relacao natural, nao artefato.
3. Os chunks marcados como "trecho" cobrem quanto da ementa?

Nao usa GPU. Roda no no de login em segundos:
    python scripts/diagnosticar_vazamento.py --config_base configs/config_base_smoke.yaml --split test
"""

from __future__ import annotations

import argparse
import difflib
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.ir_evaluator import carregar_chunks_jsonl, carregar_processos_jsonl
from src.evaluation.leakage_audit import (
    cobertura_ementa,
    normalizar_texto_auditoria,
    remover_rotulos_iniciais,
)


def carregar_yaml(caminho: Path) -> dict:
    import yaml
    with caminho.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config_base", type=Path, default=BASE_DIR / "configs" / "config_base.yaml")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--exemplos", type=int, default=3, help="Exemplos impressos por caso.")
    parser.add_argument(
        "--bruto",
        action="store_true",
        help="Le o arquivo de pares bruto (config_base) em vez dos splits. Sem chunks: secoes 5 e 6 ficam vazias.",
    )
    parser.add_argument("--amostra", type=int, default=3000, help="Com --bruto: processos sorteados do arquivo inteiro.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# Numero CNJ depois de normalizar: "0800666 57 2015 8 10 0011"
_RE_CNJ = re.compile(r"\b\d{7} \d{2} \d{4} \d \d{2} \d{4}\b")
_CABECALHO = ("sessao do dia", "recorrente", "recorrido", "advogad", "relator", "origem")


def _carregar_bruto(cfg: dict, amostra: int, seed: int) -> list:
    """Le os pares brutos com a mesma limpeza da Etapa 1 e sorteia uma amostra do arquivo inteiro."""
    from src.data.cleaner import carregar_e_filtrar_processos

    dados = cfg.get("dados", {})
    base = Path(dados.get("diretorio_base_pares", "."))
    caminhos = [base / a for a in dados.get("arquivos_pares", [])]
    processos, _ = carregar_e_filtrar_processos(
        caminho_arquivo=caminhos,
        mapeamento_colunas=dados.get("colunas", {}),
    )
    print(f"Processos validos no arquivo bruto: {len(processos):,}")
    if amostra and len(processos) > amostra:
        processos = random.Random(seed).sample(processos, amostra)
    return processos


def _faixa(valor: float, cortes: list[float]) -> str:
    for inicio, fim in zip(cortes, cortes[1:]):
        if inicio <= valor < fim:
            return f"[{inicio:.0%}, {min(fim, 1):.0%}{']' if fim > 1 else ')'}"
    return "?"


def _ordem_faixa(rotulo: str) -> float:
    # "[50%, 80%)" -> 50.0: ordena as faixas pelo numero, nao alfabeticamente
    return float(rotulo[1 : rotulo.index("%")])


def _maior_trecho_contiguo(ementa: list[str], teor: list[str]) -> tuple[int, int]:
    """(tamanho em palavras, posicao no teor) do maior trecho contiguo da ementa no teor."""
    sm = difflib.SequenceMatcher(None, ementa, teor, autojunk=False)
    m = sm.find_longest_match(0, len(ementa), 0, len(teor))
    return m.size, m.b


def _janela_comum(ementa: str, chunk: str, tamanho: int = 200) -> str:
    janelas = {chunk[i : i + tamanho] for i in range(max(0, len(chunk) - tamanho + 1))}
    for i in range(max(0, len(ementa) - tamanho + 1)):
        if ementa[i : i + tamanho] in janelas:
            return ementa[i : i + tamanho]
    return ""


def main() -> None:
    args = parse_args()
    cfg = carregar_yaml(args.config_base)
    dir_dados = args.data_dir or Path(cfg.get("dados", {}).get("diretorio_processados", "data/processed"))
    if not dir_dados.is_absolute():
        dir_dados = BASE_DIR / dir_dados

    if args.bruto:
        processos = _carregar_bruto(cfg, args.amostra, args.seed)
        chunks = []
        rotulo = f"arquivo bruto, amostra aleatoria de {len(processos):,}"
    else:
        processos = carregar_processos_jsonl(dir_dados / f"{args.split}_processos.jsonl")
        chunks = carregar_chunks_jsonl(dir_dados / f"{args.split}_corpus_chunks.jsonl")
        rotulo = f"split {args.split}"
    chunks_por_proc: dict[str, list] = defaultdict(list)
    for ch in chunks:
        chunks_por_proc[ch.processo_id].append(ch)

    saida: list[str] = []

    def p(texto: str = "") -> None:
        print(texto)
        saida.append(texto)

    ocorrencias = Counter()
    posicao_bloco = Counter()
    faixa_contiguo = Counter()
    faixa_cobertura_total = Counter()
    razoes_tamanho: list[float] = []
    palavras_ementa: list[int] = []
    cobertura_marcados = Counter()
    tamanho_chunk_na_ementa: list[int] = []
    sem_limpo_por_tipo = Counter()
    ex_frases: list[tuple] = []
    ex_bloco: list[tuple] = []
    ex_chunk_na_ementa: list[tuple] = []
    sobra_teor = Counter()
    composicao = Counter()
    ex_ementa_bruta: list[tuple] = []

    cortes = [0, 0.25, 0.5, 0.8, 0.999, 1.01]
    cortes_sobra = [0, 100, 300, 500, 1000, 10**9]
    for proc in processos:
        e_txt = remover_rotulos_iniciais(normalizar_texto_auditoria(proc.ementa))
        t_txt = normalizar_texto_auditoria(proc.inteiro_teor)
        e_pal, t_pal = e_txt.split(), t_txt.split()
        if not e_pal or not t_pal:
            continue
        palavras_ementa.append(len(e_pal))
        razoes_tamanho.append(len(e_pal) / len(t_pal))

        n_integral = t_txt.count(e_txt)
        ocorrencias["0" if n_integral == 0 else ("1" if n_integral == 1 else "2+")] += 1

        tam, pos = _maior_trecho_contiguo(e_pal, t_pal)
        frac_contiguo = tam / len(e_pal)
        faixa_contiguo[_faixa(frac_contiguo, cortes)] += 1
        if frac_contiguo >= 0.8:
            rel = pos / max(1, len(t_pal))
            posicao_bloco["inicio (primeiros 20%)" if rel < 0.2 else ("fim (ultimos 20%)" if rel > 0.8 else "meio")] += 1

        cob_total, _ = cobertura_ementa(e_pal, t_pal, 6)
        faixa_cobertura_total[_faixa(cob_total, cortes)] += 1

        # Quanto texto do teor sobra fora do bloco da ementa (o que poderia virar positivo)
        sobra = len(t_pal) - (tam if frac_contiguo >= 0.5 else 0)
        for inicio, fim in zip(cortes_sobra, cortes_sobra[1:]):
            if inicio <= sobra < fim:
                sobra_teor[(inicio, fim)] += 1
                break

        # Tipo de decisao e cabecalho dentro do campo ementa
        inicio_teor = t_txt[:3000]
        if "turma recursal" in inicio_teor or "recurso inominado" in inicio_teor:
            composicao["turma recursal / recurso inominado"] += 1
        elif "camara" in inicio_teor or "apelacao" in inicio_teor:
            composicao["camara / apelacao"] += 1
        else:
            composicao["outro"] += 1
        inicio_ementa = e_txt[:400]
        if _RE_CNJ.search(e_txt):
            composicao["(ementa contem o numero do processo)"] += 1
        if sum(rotulo_cab in inicio_ementa for rotulo_cab in _CABECALHO) >= 2:
            composicao["(ementa comeca com cabecalho: sessao/partes/advogados)"] += 1
        if len(ex_ementa_bruta) < args.exemplos:
            ex_ementa_bruta.append((proc.id, proc.ementa[:500]))

        meus = chunks_por_proc.get(proc.id, [])
        if meus and all(c.contem_vazamento_ementa for c in meus):
            sem_limpo_por_tipo["ementa em bloco (>=80% contigua)" if frac_contiguo >= 0.8 else "so frases da ementa"] += 1

        for ch in meus:
            if not ch.contem_vazamento_ementa:
                continue
            cobertura_marcados[(ch.motivo_vazamento, _faixa(ch.cobertura_ementa, [0, 0.3, 0.75, 1.01]))] += 1
            if ch.motivo_vazamento == "chunk_na_ementa":
                tamanho_chunk_na_ementa.append(len(normalizar_texto_auditoria(ch.texto).split()))
                if len(ex_chunk_na_ementa) < args.exemplos:
                    ex_chunk_na_ementa.append((proc.id, ch.chunk_idx, len(meus), ch.texto[:300]))
            if ch.motivo_vazamento == "trecho" and ch.cobertura_ementa < 0.3 and len(ex_frases) < args.exemplos:
                janela = _janela_comum(e_txt, normalizar_texto_auditoria(ch.texto))
                ex_frases.append((proc.id, ch.chunk_idx, ch.cobertura_ementa, janela))
        if frac_contiguo >= 0.8 and len(ex_bloco) < args.exemplos:
            ex_bloco.append((proc.id, pos, len(t_pal), " ".join(t_pal[max(0, pos - 25) : pos + 15])))

    n = sum(ocorrencias.values())
    p("=" * 72)
    p(f"DIAGNOSTICO DE VAZAMENTO - {rotulo} ({n} processos) - {dir_dados}")
    p("=" * 72)
    p("\n1. A ementa INTEIRA (identica, apos normalizar) aparece no teor quantas vezes?")
    for k in ("0", "1", "2+"):
        p(f"   {k:>3}: {ocorrencias[k]:>6} ({ocorrencias[k] / max(1, n):.0%})")
    p("\n2. Maior trecho CONTIGUO da ementa encontrado no teor (% das palavras da ementa):")
    for k in sorted(faixa_contiguo, key=_ordem_faixa):
        p(f"   {k:<12}: {faixa_contiguo[k]:>6} ({faixa_contiguo[k] / max(1, n):.0%})")
    p("   Onde fica esse bloco, quando cobre >= 80% da ementa:")
    for k, v in posicao_bloco.most_common():
        p(f"     - {k}: {v}")
    p("\n3. Cobertura TOTAL da ementa pelo teor (blocos de 6 palavras, contiguos ou nao):")
    for k in sorted(faixa_cobertura_total, key=_ordem_faixa):
        p(f"   {k:<12}: {faixa_cobertura_total[k]:>6} ({faixa_cobertura_total[k] / max(1, n):.0%})")
    if razoes_tamanho:
        q = statistics.quantiles(razoes_tamanho, n=4) if len(razoes_tamanho) > 3 else razoes_tamanho
        p(f"\n4. Tamanho da ementa em relacao ao teor (palavras): mediana {statistics.median(razoes_tamanho):.0%} | quartis {[f'{x:.0%}' for x in q]}")
        p(f"   Ementa: mediana de {statistics.median(palavras_ementa):.0f} palavras (max {max(palavras_ementa)})")
    p("\n5. Chunks marcados: motivo x cobertura da ementa no chunk")
    for (motivo, faixa), v in sorted(cobertura_marcados.items()):
        p(f"   {motivo:<16} {faixa:<12}: {v}")
    if tamanho_chunk_na_ementa:
        p(f"   'chunk_na_ementa': chunks com mediana de {statistics.median(tamanho_chunk_na_ementa):.0f} palavras "
          f"(min {min(tamanho_chunk_na_ementa)}, max {max(tamanho_chunk_na_ementa)})")
    p("\n6. Processos SEM nenhum chunk limpo, por tipo:")
    for k, v in sem_limpo_por_tipo.most_common():
        p(f"   - {k}: {v}")
    p("\n7. Palavras do teor FORA do bloco da ementa (~500 palavras = 1 chunk de 768 tokens):")
    for (inicio, fim) in zip(cortes_sobra, cortes_sobra[1:]):
        v = sobra_teor[(inicio, fim)]
        rotulo_faixa = f">= {inicio}" if fim > 10**8 else f"{inicio}-{fim - 1}"
        p(f"   {rotulo_faixa:<10}: {v:>6} ({v / max(1, n):.0%})")
    p("\n8. Tipo de decisao (inicio do teor) e cabecalho no campo ementa:")
    for k, v in composicao.most_common():
        p(f"   - {k}: {v} ({v / max(1, n):.0%})")

    p("\n" + "-" * 72)
    p("EXEMPLOS: chunk marcado por 'trecho' que cobre < 30% da ementa (frase em comum)")
    for pid, idx, cob, janela in ex_frases:
        p(f"\n[{pid} chunk {idx} | cobertura {cob:.0%}] janela de 200 caracteres em comum:\n  \"{janela}\"")
    p("\nEXEMPLOS: ementa em bloco (>= 80% contigua) dentro do teor - contexto antes do bloco")
    for pid, pos, total, ctx in ex_bloco:
        p(f"\n[{pid} | palavra {pos} de {total}] ...{ctx}...")
    p("\nEXEMPLOS: 'chunk_na_ementa' (chunk inteiro contido na ementa)")
    for pid, idx, total, txt in ex_chunk_na_ementa:
        p(f"\n[{pid} chunk {idx} de {total}] \"{txt}\"")
    p("\nEXEMPLOS: inicio do campo ementa, como esta no arquivo (500 caracteres)")
    for pid, txt in ex_ementa_bruta:
        p(f"\n[{pid}] \"{txt}\"")

    dir_dados.mkdir(parents=True, exist_ok=True)
    destino = dir_dados / f"diagnostico_vazamento_{'bruto' if args.bruto else args.split}.txt"
    destino.write_text("\n".join(saida) + "\n", encoding="utf-8")
    print(f"\nSalvo em: {destino}")


if __name__ == "__main__":
    main()
