#!/usr/bin/env python3
"""
Compara arquivos de pares (ementa + inteiro teor) de varios tribunais para escolher
qual serve para a tarefa "ementa -> trecho do mesmo acordao".

Arquivo AUTONOMO: usa so a biblioteca padrao do Python (3.8+). Copie este arquivo para a
pasta dos pares e rode la, sem o resto do repositorio:

    cd ~/bumbalm/Pares_Tribunais
    python3 diagnosticar_tribunais.py                      # todos os pares_*.jsonl da pasta
    python3 diagnosticar_tribunais.py pares_tjrs.jsonl pares_tjdf.jsonl --amostra 2000

Para cada arquivo:
1. Le o arquivo uma vez e sorteia N linhas (amostragem por reservatorio, sem parse).
2. Mede, nas linhas sorteadas: tamanho da ementa, quanto dela aparece copiado no teor,
   quanto texto sobra fora da ementa, tipo de decisao (Turma Recursal x Camara) e
   cabecalho/numero do processo dentro da ementa.
3. Grava diagnosticos/<arquivo>.txt com numeros e exemplos.

No fim, grava diagnosticos/resumo_tribunais.txt com uma linha por tribunal.

Uma barra de progresso mostra a leitura (em GB) e a analise (em processos).
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

# Nomes de campo usados pelos diferentes arquivos (mesma ordem de busca do cleaner do projeto)
CAMPOS_EMENTA = ("str_ementa", "ementa", "SUMARIO", "sumario")
CAMPOS_TEOR = ("str_inteiro_teor", "texto_inteiro_teor", "inteiro_teor")
CAMPOS_NUMERO = ("str_numero_unico", "numero_processo", "numero", "PROC", "KEY", "NUMACORDAO")

_RE_TAGS = re.compile(r"<[^>]+>")
_RE_NAO_ALFANUM = re.compile(r"[^a-z0-9]+")
_RE_ROTULOS = re.compile(
    r"^(?:ementa|acordao|relatorio|voto(?: do relator)?|decisao|vistos?|relatados?|discutidos?|e acordam)\s+"
)
# Numero CNJ depois de normalizar: "0800666 57 2015 8 10 0011"
_RE_CNJ = re.compile(r"\b\d{7} \d{2} \d{4} \d \d{2} \d{4}\b")
_CABECALHO = ("sessao do dia", "recorrente", "recorrido", "apelante", "apelado", "advogad", "relator", "origem")


class Barra:
    """Barra de progresso simples em uma linha (funciona no tmux). Atualiza no maximo 4x por segundo."""

    def __init__(self, total: float, descricao: str, unidade: str = "", escala: float = 1.0):
        self.total = max(total, 1e-9)
        self.descricao = descricao
        self.unidade = unidade
        self.escala = escala
        self.atual = 0.0
        self.inicio = time.time()
        self.ultimo = 0.0

    def avancar(self, quanto: float = 1.0) -> None:
        self.atual += quanto
        agora = time.time()
        if agora - self.ultimo >= 0.25 or self.atual >= self.total:
            self.ultimo = agora
            self._desenhar(agora)

    def _desenhar(self, agora: float) -> None:
        frac = min(1.0, self.atual / self.total)
        cheio = int(30 * frac)
        decorrido = agora - self.inicio
        restante = decorrido * (1 - frac) / frac if frac > 0 else 0
        valores = f"{self.atual / self.escala:,.1f}/{self.total / self.escala:,.1f}{self.unidade}"
        linha = (
            f"\r  {self.descricao:<22} [{'#' * cheio}{'.' * (30 - cheio)}] {frac:4.0%} "
            f"{valores}  {_tempo(decorrido)} decorrido, ~{_tempo(restante)} restante"
        )
        sys.stderr.write(linha[:160].ljust(120))
        sys.stderr.flush()

    def fechar(self) -> None:
        self._desenhar(time.time())
        sys.stderr.write("\n")
        sys.stderr.flush()


def _tempo(segundos: float) -> str:
    segundos = int(segundos)
    if segundos >= 3600:
        return f"{segundos // 3600}h{(segundos % 3600) // 60:02d}m"
    return f"{segundos // 60}m{segundos % 60:02d}s"


def normalizar(texto: str) -> str:
    """Tira HTML, acentos e pontuacao. Mesma normalizacao da auditoria de vazamento do projeto."""
    texto = _RE_TAGS.sub(" ", html.unescape(str(texto or "")))
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return _RE_NAO_ALFANUM.sub(" ", texto).strip()


def sem_rotulos(texto: str) -> str:
    while True:
        novo = _RE_ROTULOS.sub("", texto)
        if novo == texto:
            return texto
        texto = novo


def primeiro_campo(dado: dict, campos: tuple) -> str:
    for campo in campos:
        valor = dado.get(campo)
        if valor:
            return str(valor)
    return ""


def sortear_linhas(caminho: Path, quantidade: int, seed: int) -> tuple:
    """Amostragem por reservatorio: uma leitura do arquivo, `quantidade` linhas uniformes."""
    rng = random.Random(seed)
    escolhidas: list = []
    total = 0
    barra = Barra(caminho.stat().st_size, f"lendo {caminho.name}", " GB", 1024 ** 3)
    with caminho.open("rb") as f:
        for linha in f:
            barra.avancar(len(linha))
            if not linha.strip():
                continue
            total += 1
            if len(escolhidas) < quantidade:
                escolhidas.append(linha)
            else:
                j = rng.randrange(total)
                if j < quantidade:
                    escolhidas[j] = linha
    barra.fechar()
    return escolhidas, total


def maior_trecho_contiguo(ementa: list, teor: list) -> tuple:
    m = difflib.SequenceMatcher(None, ementa, teor, autojunk=False).find_longest_match(0, len(ementa), 0, len(teor))
    return m.size, m.b


def cobertura(ementa: list, teor: list, ngram: int = 6) -> float:
    """Fracao das palavras da ementa que aparecem no teor em blocos de `ngram` palavras."""
    if len(ementa) < ngram or len(teor) < ngram:
        return 0.0
    blocos = {tuple(teor[i : i + ngram]) for i in range(len(teor) - ngram + 1)}
    coberta = [False] * len(ementa)
    for i in range(len(ementa) - ngram + 1):
        if tuple(ementa[i : i + ngram]) in blocos:
            for j in range(i, i + ngram):
                coberta[j] = True
    return sum(coberta) / len(ementa)


def _pct(parte: int, total: int) -> str:
    return f"{parte / max(1, total):.0%}"


def analisar_arquivo(caminho: Path, amostra: int, seed: int, exemplos: int, pasta_saida: Path) -> dict:
    print(f"\n=== {caminho.name} ({caminho.stat().st_size / 1024 ** 3:.1f} GB) ===")
    # Sorteia 30% a mais: parte das linhas cai nos filtros abaixo
    linhas, total_linhas = sortear_linhas(caminho, int(amostra * 1.3), seed)

    descartes = Counter()
    registros = []
    for linha in linhas:
        try:
            dado = json.loads(linha)
        except (json.JSONDecodeError, UnicodeDecodeError):
            descartes["json_invalido"] += 1
            continue
        if not isinstance(dado, dict):
            descartes["nao_e_objeto"] += 1
            continue
        ementa = primeiro_campo(dado, CAMPOS_EMENTA).strip()
        teor = primeiro_campo(dado, CAMPOS_TEOR).strip()
        # Mesmos filtros minimos da Etapa 1 (config_data.yaml)
        if not ementa or not teor:
            descartes["campo_vazio"] += 1
        elif len(ementa) < 50:
            descartes["ementa_curta"] += 1
        elif len(teor) < 200:
            descartes["teor_curto"] += 1
        elif len(teor) < len(ementa):
            descartes["teor_menor_que_ementa"] += 1
        else:
            registros.append((primeiro_campo(dado, CAMPOS_NUMERO), ementa, teor))
    if len(registros) > amostra:
        registros = random.Random(seed).sample(registros, amostra)
    n = len(registros)

    faixa_contiguo = Counter()
    posicao_bloco = Counter()
    sobra = Counter()
    composicao = Counter()
    razoes: list = []
    palavras_ementa: list = []
    palavras_teor: list = []
    coberturas: list = []
    ex_ementa: list = []
    ex_bloco: list = []
    faixas_sobra = [(0, 100), (100, 300), (300, 500), (500, 1000), (1000, 10 ** 9)]

    barra = Barra(n, "analisando", " proc")
    for numero, ementa, teor in registros:
        barra.avancar()
        e_txt = sem_rotulos(normalizar(ementa))
        t_txt = normalizar(teor)
        e_pal, t_pal = e_txt.split(), t_txt.split()
        if not e_pal or not t_pal:
            continue
        palavras_ementa.append(len(e_pal))
        palavras_teor.append(len(t_pal))
        razoes.append(len(e_pal) / len(t_pal))
        coberturas.append(cobertura(e_pal, t_pal))

        tam, pos = maior_trecho_contiguo(e_pal, t_pal)
        frac = tam / len(e_pal)
        faixa_contiguo["< 50%" if frac < 0.5 else ("50-80%" if frac < 0.8 else ">= 80%")] += 1
        if frac >= 0.8:
            rel = pos / len(t_pal)
            posicao_bloco["inicio" if rel < 0.2 else ("fim" if rel > 0.8 else "meio")] += 1
            if len(ex_bloco) < exemplos:
                ex_bloco.append((numero, pos, len(t_pal), " ".join(t_pal[max(0, pos - 20) : pos + 12])))

        fora = len(t_pal) - (tam if frac >= 0.5 else 0)
        for inicio, fim in faixas_sobra:
            if inicio <= fora < fim:
                sobra[(inicio, fim)] += 1
                break

        inicio_teor = t_txt[:3000]
        if "turma recursal" in inicio_teor or "recurso inominado" in inicio_teor or "juizado especial" in inicio_teor:
            composicao["turma_recursal"] += 1
        elif "camara" in inicio_teor or "apelacao" in inicio_teor or "apelante" in inicio_teor:
            composicao["camara_apelacao"] += 1
        else:
            composicao["outro"] += 1
        if _RE_CNJ.search(e_txt):
            composicao["numero_na_ementa"] += 1
        if sum(rot in e_txt[:400] for rot in _CABECALHO) >= 2:
            composicao["cabecalho_na_ementa"] += 1
        if len(ex_ementa) < exemplos:
            ex_ementa.append((numero, ementa[:400]))
    barra.fechar()

    m = len(palavras_ementa)
    mediana = lambda xs: statistics.median(xs) if xs else 0  # noqa: E731
    quartis = statistics.quantiles(razoes, n=4) if len(razoes) > 3 else razoes
    fora_menos_500 = sum(sobra[f] for f in faixas_sobra[:3])
    fora_1000 = sobra[faixas_sobra[4]]

    linhas_rel = [
        "=" * 72,
        f"DIAGNOSTICO - {caminho.name}",
        "=" * 72,
        f"Linhas no arquivo: {total_linhas:,} | sorteadas: {len(linhas):,} | analisadas: {m:,}",
        f"Descartes (filtros da Etapa 1): {dict(descartes)}",
        "",
        "1. Tamanho (palavras, mediana)",
        f"   ementa: {mediana(palavras_ementa):.0f} (max {max(palavras_ementa, default=0)}) | "
        f"teor: {mediana(palavras_teor):.0f}",
        f"   ementa / teor: mediana {mediana(razoes):.0%} | quartis {[f'{x:.0%}' for x in quartis]}",
        "",
        "2. Maior trecho CONTIGUO da ementa dentro do teor",
    ]
    for k in ("< 50%", "50-80%", ">= 80%"):
        linhas_rel.append(f"   {k:<7}: {faixa_contiguo[k]:>6} ({_pct(faixa_contiguo[k], m)})")
    linhas_rel.append(f"   onde fica o bloco (>= 80%): {dict(posicao_bloco)}")
    linhas_rel.append(f"   cobertura total da ementa pelo teor (mediana): {mediana(coberturas):.0%}")
    linhas_rel += ["", "3. Palavras do teor FORA do bloco da ementa (~500 palavras = 1 chunk de 768 tokens)"]
    for inicio, fim in faixas_sobra:
        rot = f">= {inicio}" if fim > 10 ** 8 else f"{inicio}-{fim - 1}"
        linhas_rel.append(f"   {rot:<9}: {sobra[(inicio, fim)]:>6} ({_pct(sobra[(inicio, fim)], m)})")
    linhas_rel += ["", "4. Tipo de decisao (inicio do teor) e cabecalho na ementa"]
    for k in ("turma_recursal", "camara_apelacao", "outro", "numero_na_ementa", "cabecalho_na_ementa"):
        linhas_rel.append(f"   {k:<20}: {composicao[k]:>6} ({_pct(composicao[k], m)})")
    linhas_rel += ["", "-" * 72, "EXEMPLOS: inicio do campo ementa (400 caracteres)"]
    for numero, txt in ex_ementa:
        linhas_rel.append(f"\n[{numero}] {txt!r}")
    linhas_rel.append("\nEXEMPLOS: contexto antes do bloco da ementa dentro do teor")
    for numero, pos, total, ctx in ex_bloco:
        linhas_rel.append(f"\n[{numero} | palavra {pos} de {total}] ...{ctx}...")

    pasta_saida.mkdir(parents=True, exist_ok=True)
    destino = pasta_saida / f"{caminho.stem}.txt"
    destino.write_text("\n".join(linhas_rel) + "\n", encoding="utf-8")
    print("\n".join(linhas_rel[:30]))
    print(f"\nRelatorio completo: {destino}")

    return {
        "arquivo": caminho.stem.replace("pares_", "").upper(),
        "gb": caminho.stat().st_size / 1024 ** 3,
        "linhas": total_linhas,
        "n": m,
        "ementa_pal": mediana(palavras_ementa),
        "teor_pal": mediana(palavras_teor),
        "razao": mediana(razoes),
        "fora_menos_500": fora_menos_500 / max(1, m),
        "fora_1000": fora_1000 / max(1, m),
        "tr": composicao["turma_recursal"] / max(1, m),
        "camara": composicao["camara_apelacao"] / max(1, m),
        "numero": composicao["numero_na_ementa"] / max(1, m),
        "cabecalho": composicao["cabecalho_na_ementa"] / max(1, m),
    }


def gravar_resumo(resultados: list, pasta_saida: Path, imprimir: bool = True) -> None:
    cab = (
        f"{'Tribunal':<9}{'GB':>6}{'Linhas':>11}{'Ementa':>8}{'Teor':>7}{'Em/Teor':>9}"
        f"{'Fora<500':>10}{'Fora>=1k':>10}{'TurmaRec':>10}{'Camara':>8}{'NumEmenta':>11}{'Cabecalho':>11}"
    )
    linhas = [
        "RESUMO POR TRIBUNAL (medianas em palavras; demais colunas em % dos processos analisados)",
        "Bom para 'ementa -> trecho': Em/Teor baixo, Fora>=1k alto, NumEmenta e Cabecalho baixos.",
        "",
        cab,
        "-" * len(cab),
    ]
    ordenados = sorted(resultados, key=lambda r: (-r["fora_1000"], r["razao"]))
    for r in ordenados:
        linhas.append(
            f"{r['arquivo']:<9}{r['gb']:>6.1f}{r['linhas']:>11,}{r['ementa_pal']:>8.0f}{r['teor_pal']:>7.0f}"
            f"{r['razao']:>9.0%}{r['fora_menos_500']:>10.0%}{r['fora_1000']:>10.0%}{r['tr']:>10.0%}"
            f"{r['camara']:>8.0%}{r['numero']:>11.0%}{r['cabecalho']:>11.0%}"
        )
    pasta_saida.mkdir(parents=True, exist_ok=True)
    destino = pasta_saida / "resumo_tribunais.txt"
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    if imprimir:
        print("\n" + "\n".join(linhas))
        print(f"\nResumo salvo em: {destino}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arquivos", nargs="*", type=Path, help="Arquivos de pares. Padrao: pares_*.jsonl da pasta atual.")
    parser.add_argument("--amostra", type=int, default=2000, help="Processos sorteados por arquivo.")
    parser.add_argument("--exemplos", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saida", type=Path, default=Path("diagnosticos"))
    args = parser.parse_args()

    arquivos = args.arquivos or sorted(Path(".").glob("pares_*.jsonl"))
    if not arquivos:
        sys.exit("Nenhum arquivo pares_*.jsonl encontrado. Rode dentro da pasta dos pares ou passe os caminhos.")
    print(f"{len(arquivos)} arquivo(s), amostra de {args.amostra:,} processos por arquivo. Relatorios em: {args.saida}/")

    resultados = []
    inicio = time.time()
    for i, caminho in enumerate(arquivos, start=1):
        print(f"\n[{i}/{len(arquivos)}] decorrido: {_tempo(time.time() - inicio)}")
        try:
            resultados.append(analisar_arquivo(caminho, args.amostra, args.seed, args.exemplos, args.saida))
        except Exception as exc:  # um arquivo com problema nao derruba a comparacao inteira
            print(f"\nERRO em {caminho.name}: {exc!r}. Seguindo para o proximo.")
        # Resumo parcial a cada arquivo: se a sessao cair, o que ja rodou nao se perde
        gravar_resumo(resultados, args.saida, imprimir=False)
    gravar_resumo(resultados, args.saida)
    print(f"Tempo total: {_tempo(time.time() - inicio)}")


if __name__ == "__main__":
    main()
