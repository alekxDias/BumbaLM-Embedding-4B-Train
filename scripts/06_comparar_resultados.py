"""
BumbaLM - Etapa 6: tabela final e testes de significancia pareados

Junta os resultados de avaliacao no teste (um por modelo) e monta o que entra no capitulo de
resultados do TCC:

1. Tabela com as metricas de cada modelo (das medias gravadas pela Etapa 5 ou pelo BM25).
2. A mesma tabela separada entre consultas sem e com "gemeos" (quase duplicatas relevantes), quando
   algum arquivo por consulta traz o campo tem_gemeos.
3. Testes pareados por consulta para cada par pedido, em nDCG@10 e MRR@10:
   - diferenca das medias com intervalo de confianca de 95% por bootstrap pareado;
   - p-valor bilateral pelo teste de aleatorizacao pareado (troca de sinal), o teste recomendado
     para sistemas de recuperacao (Smucker, Allan e Carterette, 2007);
   - p-valor ajustado por Holm-Bonferroni dentro de cada metrica;
   - quantas consultas cada modelo ganha, empata e perde.
4. Curvas de validacao do treino (nDCG@10 e MRR@10 por passo), se os CSVs forem passados.

Cada resultado e uma pasta com metricas_<split>_<nome>.json e por_consulta_<split>_<nome>.jsonl, como
as que a Etapa 5 grava em avaliacoes/test/<modelo>/. Todas precisam ter exatamente as mesmas consultas.

Uso:
    python scripts/06_comparar_resultados.py \
        --resultado base=avaliacoes/test/baseline --resultado bm25=avaliacao_bm25 \
        --resultado nenhum=avaliacoes/test/nenhum --resultado dificeis=avaliacoes/test/dificeis \
        --pares dificeis:nenhum,dificeis:base,nenhum:base,dificeis:bm25,nenhum:bm25,base:bm25 \
        --curva nenhum=modelos/nenhum/metricas_treino_val.csv --curva dificeis=modelos/dificeis/metricas_treino_val.csv \
        --saida resultados_finais
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

METRICAS_TABELA = ["ndcg@10", "mrr@10", "map@100", "accuracy@1", "accuracy@5", "accuracy@10", "recall@10"]
METRICAS_TESTE = ["ndcg@10", "mrr@10"]
ROTULOS = {"ndcg@10": "nDCG@10", "mrr@10": "MRR@10"}


# ----------------------------------------------------------------------------------------------
# Leitura
# ----------------------------------------------------------------------------------------------
def _unico(pasta: Path, padrao: str) -> Path:
    achados = sorted(pasta.glob(padrao))
    if len(achados) != 1:
        raise FileNotFoundError(
            f"Esperado exatamente 1 arquivo '{padrao}' em {pasta}, achei {len(achados)}: {[a.name for a in achados]}"
        )
    return achados[0]


def carregar_resultado(nome: str, caminho: Path) -> dict[str, Any]:
    """Le as metricas globais e o arquivo por consulta de uma pasta de resultado."""
    pasta = Path(caminho)
    if not pasta.is_dir():
        raise FileNotFoundError(f"Resultado '{nome}': pasta {pasta} nao existe.")
    arq_metricas = _unico(pasta, "metricas_*.json")
    arq_consultas = _unico(pasta, "por_consulta_*.jsonl")
    metricas = json.loads(arq_metricas.read_text(encoding="utf-8"))
    consultas: dict[str, dict[str, Any]] = {}
    with arq_consultas.open(encoding="utf-8") as f:
        for linha in f:
            if linha.strip():
                registro = json.loads(linha)
                consultas[registro["query_id"]] = registro
    return {
        "nome": nome,
        "pasta": str(pasta),
        "metricas_global": metricas.get("metricas_global", {}),
        "protocolo": metricas.get("protocolo", {}),
        "configuracao": metricas.get("configuracao_avaliacao", {}),
        "consultas": consultas,
    }


def conferir_mesmas_consultas(resultados: dict[str, dict[str, Any]]) -> list[str]:
    """Para se algum resultado tiver consultas diferentes. Devolve os ids na ordem do primeiro."""
    nomes = list(resultados)
    referencia = resultados[nomes[0]]["consultas"]
    for nome in nomes[1:]:
        atuais = resultados[nome]["consultas"]
        if set(atuais) != set(referencia):
            so_ref = len(set(referencia) - set(atuais))
            so_atual = len(set(atuais) - set(referencia))
            raise ValueError(
                f"'{nome}' e '{nomes[0]}' nao tem as mesmas consultas ({so_ref} so em '{nomes[0]}', "
                f"{so_atual} so em '{nome}'). Os resultados nao sao do mesmo teste."
            )
    return list(referencia)


def vetor(resultado: dict[str, Any], ids: list[str], metrica: str) -> np.ndarray:
    return np.array([float(resultado["consultas"][q][metrica]) for q in ids], dtype=np.float64)


def gemeos_por_consulta(resultados: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, bool] | None:
    """tem_gemeos vem do arquivo por consulta do BM25 (e o mesmo para todos os modelos)."""
    for resultado in resultados.values():
        amostra = next(iter(resultado["consultas"].values()))
        if "tem_gemeos" in amostra:
            return {q: bool(resultado["consultas"][q]["tem_gemeos"]) for q in ids}
    return None


# ----------------------------------------------------------------------------------------------
# Estatistica
# ----------------------------------------------------------------------------------------------
def teste_aleatorizacao_pareado(a: np.ndarray, b: np.ndarray, n_permutacoes: int, semente: int,
                                bloco: int = 500) -> float:
    """p-valor bilateral: com que frequencia trocar ao acaso o sinal das diferencas da uma media tao extrema."""
    d = a - b
    observado = abs(d.mean())
    rng = np.random.default_rng(semente)
    extremos = 0
    feitos = 0
    while feitos < n_permutacoes:
        k = min(bloco, n_permutacoes - feitos)
        sinais = rng.integers(0, 2, size=(k, d.size), dtype=np.int8) * 2 - 1
        medias = np.abs((sinais * d).mean(axis=1))
        # Tolerancia para empates numericos com o valor observado
        extremos += int(np.count_nonzero(medias >= observado - 1e-12))
        feitos += k
    return (extremos + 1) / (n_permutacoes + 1)


def ic_bootstrap_pareado(a: np.ndarray, b: np.ndarray, n_reamostras: int, semente: int,
                         bloco: int = 500) -> tuple[float, float]:
    """Intervalo de 95% (percentis 2,5 e 97,5) da diferenca das medias, reamostrando consultas."""
    d = a - b
    rng = np.random.default_rng(semente)
    medias = np.empty(n_reamostras, dtype=np.float64)
    feitos = 0
    while feitos < n_reamostras:
        k = min(bloco, n_reamostras - feitos)
        indices = rng.integers(0, d.size, size=(k, d.size))
        medias[feitos:feitos + k] = d[indices].mean(axis=1)
        feitos += k
    baixo, alto = np.percentile(medias, [2.5, 97.5])
    return float(baixo), float(alto)


def holm(p_valores: list[float]) -> list[float]:
    """Ajuste de Holm-Bonferroni: controla a chance de qualquer falso positivo entre as comparacoes."""
    m = len(p_valores)
    ordem = sorted(range(m), key=lambda i: p_valores[i])
    ajustados = [0.0] * m
    maximo = 0.0
    for posicao, i in enumerate(ordem):
        valor = min(1.0, (m - posicao) * p_valores[i])
        maximo = max(maximo, valor)
        ajustados[i] = maximo
    return ajustados


def comparar_pares(resultados: dict[str, dict[str, Any]], ids: list[str], pares: list[tuple[str, str]],
                   n_permutacoes: int, n_bootstrap: int, semente: int) -> list[dict[str, Any]]:
    linhas: list[dict[str, Any]] = []
    for metrica in METRICAS_TESTE:
        bloco: list[dict[str, Any]] = []
        for indice, (nome_a, nome_b) in enumerate(pares):
            a = vetor(resultados[nome_a], ids, metrica)
            b = vetor(resultados[nome_b], ids, metrica)
            # Semente diferente por par e metrica, mas fixa: o resultado se repete a cada execucao
            s = semente + 1000 * METRICAS_TESTE.index(metrica) + indice
            p = teste_aleatorizacao_pareado(a, b, n_permutacoes, s)
            baixo, alto = ic_bootstrap_pareado(a, b, n_bootstrap, s)
            bloco.append({
                "metrica": metrica,
                "a": nome_a,
                "b": nome_b,
                "media_a": float(a.mean()),
                "media_b": float(b.mean()),
                "diferenca": float((a - b).mean()),
                "ic95_baixo": baixo,
                "ic95_alto": alto,
                "p_valor": p,
                "a_ganha": int(np.count_nonzero(a > b)),
                "empates": int(np.count_nonzero(a == b)),
                "b_ganha": int(np.count_nonzero(a < b)),
                "consultas": len(ids),
            })
        for linha, ajustado in zip(bloco, holm([l["p_valor"] for l in bloco])):
            linha["p_holm"] = ajustado
        linhas.extend(bloco)
    return linhas


# ----------------------------------------------------------------------------------------------
# Saidas
# ----------------------------------------------------------------------------------------------
def _fmt(valor: Any, casas: int = 4) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "-"
    return f"{valor:.{casas}f}"


def _fmt_p(p: float) -> str:
    return "< 0,0001" if p < 1e-4 else f"{p:.4f}".replace(".", ",")


def tabela_metricas(resultados: dict[str, dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    """Medias gravadas por cada avaliacao, conferidas contra a media dos arquivos por consulta."""
    linhas = []
    for nome, resultado in resultados.items():
        linha: dict[str, Any] = {"modelo": nome}
        for metrica in METRICAS_TABELA:
            gravada = resultado["metricas_global"].get(metrica)
            amostra = next(iter(resultado["consultas"].values()))
            recalculada = float(vetor(resultado, ids, metrica).mean()) if metrica in amostra else None
            if gravada is not None and recalculada is not None and abs(gravada - recalculada) > 1e-6:
                print(f"  AVISO: {nome} {metrica}: media gravada {gravada:.6f}, media por consulta {recalculada:.6f}")
            linha[metrica] = gravada if gravada is not None else recalculada
        linhas.append(linha)
    return linhas


def tabela_gemeos(resultados: dict[str, dict[str, Any]], ids: list[str], gemeos: dict[str, bool]) -> list[dict[str, Any]]:
    sem = [q for q in ids if not gemeos[q]]
    com = [q for q in ids if gemeos[q]]
    linhas = []
    for nome, resultado in resultados.items():
        linha: dict[str, Any] = {"modelo": nome}
        for metrica in METRICAS_TESTE:
            linha[f"{metrica} todas"] = float(vetor(resultado, ids, metrica).mean())
            linha[f"{metrica} sem gemeos"] = float(vetor(resultado, sem, metrica).mean()) if sem else float("nan")
            linha[f"{metrica} com gemeos"] = float(vetor(resultado, com, metrica).mean()) if com else float("nan")
        linhas.append(linha)
    return linhas


def gravar_csv(linhas: list[dict[str, Any]], caminho: Path) -> None:
    if not linhas:
        return
    with caminho.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def markdown_tabela(linhas: list[dict[str, Any]], colunas: list[str]) -> str:
    cabecalho = "| " + " | ".join(colunas) + " |"
    separador = "|" + "|".join("---" for _ in colunas) + "|"
    corpo = []
    for linha in linhas:
        celulas = [str(linha[c]) if isinstance(linha[c], str) else _fmt(linha[c]) for c in colunas]
        corpo.append("| " + " | ".join(celulas) + " |")
    return "\n".join([cabecalho, separador, *corpo])


def ler_curva(caminho: Path) -> dict[str, list[tuple[int, float]]]:
    """Passo e metricas de validacao do metricas_treino_val.csv (linhas sem validacao sao ignoradas)."""
    curvas: dict[str, list[tuple[int, float]]] = {"ndcg@10": [], "mrr@10": []}
    with caminho.open(encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            passo = linha.get("step") or linha.get("passo")
            if passo in (None, ""):
                continue
            for metrica in curvas:
                coluna = next((c for c in linha if c.endswith(metrica) and "val" in c and linha[c] not in ("", None)), None)
                if coluna:
                    curvas[metrica].append((int(float(passo)), float(linha[coluna])))
    for metrica in curvas:
        curvas[metrica] = sorted(set(curvas[metrica]))
    return curvas


def desenhar_curvas(curvas: dict[str, dict[str, list[tuple[int, float]]]], saida: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib nao instalado: curvas gravadas so em CSV.")
        return []
    figura, eixos = plt.subplots(1, 2, figsize=(10, 3.8))
    for eixo, metrica in zip(eixos, ["ndcg@10", "mrr@10"]):
        for nome, por_metrica in curvas.items():
            pontos = por_metrica.get(metrica, [])
            if pontos:
                eixo.plot([p for p, _ in pontos], [v for _, v in pontos], marker="o", markersize=3, label=nome)
        eixo.set_xlabel("Passo de treino")
        eixo.set_ylabel(f"{ROTULOS.get(metrica, metrica)} na validação")
        eixo.grid(alpha=0.3)
        eixo.legend()
    figura.tight_layout()
    arquivos = []
    for ext in ("png", "pdf"):
        caminho = saida / f"curvas_validacao.{ext}"
        figura.savefig(caminho, dpi=200)
        arquivos.append(caminho)
    plt.close(figura)
    return arquivos


# ----------------------------------------------------------------------------------------------
def _pares_nome_caminho(itens: list[str], opcao: str) -> dict[str, Path]:
    saida: dict[str, Path] = {}
    for item in itens:
        if "=" not in item:
            raise ValueError(f"{opcao} espera NOME=CAMINHO, recebeu: {item}")
        nome, caminho = item.split("=", 1)
        saida[nome.strip()] = Path(caminho.strip())
    return saida


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resultado", action="append", default=[], metavar="NOME=PASTA", required=True,
                        help="Pasta com metricas_*.json e por_consulta_*.jsonl. Repita para cada modelo.")
    parser.add_argument("--pares", type=str, default="",
                        help="Comparacoes A:B separadas por virgula (padrao: todos os pares, na ordem dada).")
    parser.add_argument("--curva", action="append", default=[], metavar="NOME=CSV",
                        help="metricas_treino_val.csv de uma variante, para as curvas de validacao.")
    parser.add_argument("--saida", type=Path, required=True, help="Pasta onde gravar tabelas, testes e figuras.")
    parser.add_argument("--permutacoes", type=int, default=10000)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--semente", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    caminhos = _pares_nome_caminho(args.resultado, "--resultado")
    resultados = {nome: carregar_resultado(nome, caminho) for nome, caminho in caminhos.items()}
    ids = conferir_mesmas_consultas(resultados)
    print(f"{len(resultados)} resultados com as mesmas {len(ids):,} consultas: {', '.join(resultados)}")
    for nome, r in resultados.items():
        cfg = r["configuracao"]
        detalhe = f"precisao {cfg.get('precisao', '-')}, {cfg.get('max_seq_length', '-')} tokens" if cfg else "sem configuracao gravada"
        print(f"  - {nome:<16} {detalhe} | {r['pasta']}")

    if args.pares:
        pares = []
        for item in args.pares.split(","):
            a, b = (x.strip() for x in item.split(":"))
            for n in (a, b):
                if n not in resultados:
                    raise ValueError(f"Par {item}: '{n}' nao esta entre os resultados ({', '.join(resultados)}).")
            pares.append((a, b))
    else:
        nomes = list(resultados)
        pares = [(nomes[i], nomes[j]) for i in range(len(nomes)) for j in range(i + 1, len(nomes))]

    # 1. Tabela principal
    linhas = tabela_metricas(resultados, ids)
    gravar_csv(linhas, args.saida / "tabela_metricas.csv")
    texto_md = ["## Metricas no teste", "", markdown_tabela(linhas, ["modelo", *METRICAS_TABELA]), ""]

    # 2. Sem e com gemeos
    gemeos = gemeos_por_consulta(resultados, ids)
    if gemeos:
        n_com = sum(gemeos.values())
        linhas_g = tabela_gemeos(resultados, ids, gemeos)
        gravar_csv(linhas_g, args.saida / "tabela_gemeos.csv")
        colunas_g = ["modelo"] + [c for c in linhas_g[0] if c != "modelo"]
        texto_md += [f"## Consultas sem e com gemeos ({len(ids) - n_com:,} sem, {n_com:,} com)", "",
                     markdown_tabela(linhas_g, colunas_g), ""]

    # 3. Testes pareados
    print(f"\nTestes pareados: {len(pares)} pares x {len(METRICAS_TESTE)} metricas "
          f"({args.permutacoes:,} permutacoes, {args.bootstrap:,} reamostras)...")
    testes = comparar_pares(resultados, ids, pares, args.permutacoes, args.bootstrap, args.semente)
    gravar_csv(testes, args.saida / "testes_pareados.csv")
    texto_md += ["## Testes pareados (aleatorizacao com troca de sinal; IC 95% por bootstrap; Holm por metrica)", "",
                 "| metrica | A | B | media A | media B | A - B | IC 95% | p | p Holm | A ganha / empata / B ganha |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
    for t in testes:
        texto_md.append(
            f"| {t['metrica']} | {t['a']} | {t['b']} | {_fmt(t['media_a'])} | {_fmt(t['media_b'])} | "
            f"{t['diferenca']:+.4f} | [{t['ic95_baixo']:+.4f}; {t['ic95_alto']:+.4f}] | {_fmt_p(t['p_valor'])} | "
            f"{_fmt_p(t['p_holm'])} | {t['a_ganha']:,} / {t['empates']:,} / {t['b_ganha']:,} |"
        )
    texto_md.append("")

    # 4. Curvas de validacao
    curvas_csv = _pares_nome_caminho(args.curva, "--curva")
    curvas = {nome: ler_curva(caminho) for nome, caminho in curvas_csv.items() if caminho.is_file()}
    for nome, caminho in curvas_csv.items():
        if not caminho.is_file():
            print(f"  Curva de '{nome}' nao encontrada em {caminho}: fica de fora.")
    figuras: list[Path] = []
    if curvas:
        linhas_c = [{"variante": nome, "passo": passo, "metrica": metrica, "valor": valor}
                    for nome, por_metrica in curvas.items() for metrica, pontos in por_metrica.items()
                    for passo, valor in pontos]
        gravar_csv(linhas_c, args.saida / "curvas_validacao.csv")
        figuras = desenhar_curvas(curvas, args.saida)
        texto_md += ["## Validacao durante o treino (nDCG@10)", ""]
        for nome, por_metrica in curvas.items():
            pontos = por_metrica.get("ndcg@10", [])
            if pontos:
                melhor = max(pontos, key=lambda p: p[1])
                texto_md.append(f"- {nome}: passo 0 = {_fmt(pontos[0][1])}, melhor = {_fmt(melhor[1])} "
                                f"no passo {melhor[0]}, ultimo = {_fmt(pontos[-1][1])} no passo {pontos[-1][0]}")
        texto_md.append("")

    (args.saida / "resultados.md").write_text("\n".join(texto_md), encoding="utf-8")
    resumo = {
        "consultas": len(ids),
        "resultados": {n: {"pasta": r["pasta"], "configuracao": r["configuracao"], "protocolo": r["protocolo"],
                           "metricas_global": r["metricas_global"]} for n, r in resultados.items()},
        "tabela": linhas,
        "testes_pareados": testes,
        "parametros": {"permutacoes": args.permutacoes, "bootstrap": args.bootstrap, "semente": args.semente},
    }
    (args.saida / "resultados.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "\n".join(texto_md))
    print(f"Gravado em {args.saida}: resultados.md, resultados.json, tabela_metricas.csv, testes_pareados.csv"
          + (", tabela_gemeos.csv" if gemeos else "") + "".join(f", {f.name}" for f in figuras))


if __name__ == "__main__":
    main()
