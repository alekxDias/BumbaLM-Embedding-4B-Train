#!/usr/bin/env python3
"""
Sorteia uma amostra reproduzivel de um arquivo de pares grande (ex.: pares_tjrs.jsonl, 11 GB)
para caber no tempo de GPU da V100.

Arquivo AUTONOMO: usa so a biblioteca padrao do Python (3.8+). Copie para a pasta dos pares:

    cd ~/bumbalm/Pares_Tribunais
    python3 amostrar_pares.py pares_tjrs.jsonl --n 30000
    python3 amostrar_pares.py pares_tjrs.jsonl --n 30000 --somente_camaras

O que faz:
1. Le o arquivo uma vez (barra de progresso em GB) e sorteia linhas por reservatorio,
   com um pouco de folga para os filtros.
2. Aplica os filtros minimos da Etapa 1 (ementa e teor presentes, tamanhos minimos,
   teor maior que a ementa) e, se pedido, mantem so acordaos de Camara.
3. Grava as N linhas escolhidas SEM alteracao, na ordem em que estao no arquivo original:
   pares_tjrs_amostra30k.jsonl
4. Grava um manifesto (.manifest.json) com semente, contagens, composicao e SHA-256,
   para citar no TCC e conferir a copia para o tapajos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

CAMPOS_EMENTA = ("str_ementa", "ementa", "SUMARIO", "sumario")
CAMPOS_TEOR = ("str_inteiro_teor", "texto_inteiro_teor", "inteiro_teor")
CAMPOS_NUMERO = ("str_numero_unico", "numero_processo", "numero", "PROC", "KEY", "NUMACORDAO")


class Barra:
    """Barra de progresso em uma linha (funciona no tmux)."""

    def __init__(self, total: float, descricao: str, unidade: str = "", escala: float = 1.0):
        self.total, self.descricao, self.unidade, self.escala = max(total, 1e-9), descricao, unidade, escala
        self.atual, self.inicio, self.ultimo = 0.0, time.time(), 0.0

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
        linha = (
            f"\r  {self.descricao:<24} [{'#' * cheio}{'.' * (30 - cheio)}] {frac:4.0%} "
            f"{self.atual / self.escala:,.1f}/{self.total / self.escala:,.1f}{self.unidade}  "
            f"{_tempo(decorrido)} decorrido, ~{_tempo(restante)} restante"
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


def _primeiro(dado: dict, campos: tuple) -> str:
    for campo in campos:
        if dado.get(campo):
            return str(dado[campo])
    return ""


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto or "").lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def _tipo_decisao(teor: str) -> str:
    inicio = _normalizar(teor[:4000])
    if "turma recursal" in inicio or "recurso inominado" in inicio or "juizado especial" in inicio:
        return "turma_recursal"
    if "camara" in inicio or "apelacao" in inicio or "apelante" in inicio:
        return "camara"
    return "outro"


def _codigo_tribunal(numero: str) -> str:
    # CNJ NNNNNNN-DD.AAAA.J.TR.OOOO -> "J.TR" (8.21 = TJRS, 8.20 = TJRN, 8.10 = TJMA)
    digitos = re.sub(r"\D", "", numero)
    return f"{digitos[13]}.{digitos[14:16]}" if len(digitos) == 20 else "fora do padrao CNJ"


def _motivo_descarte(ementa: str, teor: str) -> str:
    # Mesmos filtros minimos de configs/config_data.yaml
    if not ementa or not teor:
        return "campo_vazio"
    if len(ementa) < 50:
        return "ementa_curta"
    if len(teor) < 200:
        return "teor_curto"
    if len(teor) < len(ementa):
        return "teor_menor_que_ementa"
    return ""


def sortear(caminho: Path, quantidade: int, seed: int) -> tuple:
    """Reservatorio sobre as linhas brutas. Devolve [(indice_da_linha, bytes)] e o total de linhas."""
    rng = random.Random(seed)
    escolhidas: list = []
    total = 0
    barra = Barra(caminho.stat().st_size, f"lendo {caminho.name}", " GB", 1024 ** 3)
    with caminho.open("rb") as f:
        for linha in f:
            barra.avancar(len(linha))
            if not linha.strip():
                continue
            if len(escolhidas) < quantidade:
                escolhidas.append((total, linha))
            else:
                j = rng.randrange(total + 1)
                if j < quantidade:
                    escolhidas[j] = (total, linha)
            total += 1
    barra.fechar()
    return escolhidas, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arquivo", type=Path, help="Arquivo de pares de origem (ex.: pares_tjrs.jsonl).")
    parser.add_argument("--n", type=int, default=30000, help="Processos na amostra (padrao: 30000).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--somente_camaras", action="store_true", help="Descarta Turmas Recursais e 'outro'.")
    parser.add_argument("--saida", type=Path, default=None, help="Padrao: pares_<tribunal>_amostra<N>k.jsonl")
    args = parser.parse_args()

    origem = args.arquivo
    tribunal = origem.stem.replace("pares_", "").split("_")[0]
    sufixo = f"amostra{args.n // 1000}k" + ("_camaras" if args.somente_camaras else "")
    destino = args.saida or origem.with_name(f"pares_{tribunal}_{sufixo}.jsonl")
    if destino.resolve() == origem.resolve():
        sys.exit("A saida nao pode ser o proprio arquivo de origem.")

    # Folga para os filtros: ~79% do TJRS e Camara; os filtros minimos quase nao descartam.
    folga = 1.5 if args.somente_camaras else 1.15
    print(f"Origem : {origem} ({origem.stat().st_size / 1024 ** 3:.1f} GB)")
    print(f"Destino: {destino}")
    print(f"Amostra: {args.n:,} processos | semente {args.seed} | so Camaras: {args.somente_camaras}")
    inicio = time.time()

    sorteadas, total_linhas = sortear(origem, int(args.n * folga), args.seed)

    descartes = Counter()
    aceitas = []
    barra = Barra(len(sorteadas), "filtrando", " linhas")
    for indice, linha in sorteadas:
        barra.avancar()
        try:
            dado = json.loads(linha)
        except (json.JSONDecodeError, UnicodeDecodeError):
            descartes["json_invalido"] += 1
            continue
        if not isinstance(dado, dict):
            descartes["nao_e_objeto"] += 1
            continue
        ementa = _primeiro(dado, CAMPOS_EMENTA).strip()
        teor = _primeiro(dado, CAMPOS_TEOR).strip()
        motivo = _motivo_descarte(ementa, teor)
        if motivo:
            descartes[motivo] += 1
            continue
        tipo = _tipo_decisao(teor)
        if args.somente_camaras and tipo != "camara":
            descartes[f"fora_de_camara ({tipo})"] += 1
            continue
        aceitas.append((indice, linha, tipo, _codigo_tribunal(_primeiro(dado, CAMPOS_NUMERO))))
    barra.fechar()

    if len(aceitas) < args.n:
        print(f"\nAVISO: so {len(aceitas):,} linhas passaram nos filtros (pedido: {args.n:,}). Gravando todas.")
    else:
        aceitas = random.Random(args.seed).sample(aceitas, args.n)
    aceitas.sort(key=lambda item: item[0])  # ordem do arquivo original

    sha = hashlib.sha256()
    temporario = destino.with_name(destino.name + ".tmp")
    barra = Barra(len(aceitas), "gravando", " linhas")
    with temporario.open("wb") as f:
        for _, linha, _, _ in aceitas:
            linha = linha if linha.endswith(b"\n") else linha + b"\n"
            f.write(linha)
            sha.update(linha)
            barra.avancar()
    barra.fechar()
    temporario.replace(destino)

    manifesto = {
        "origem": str(origem.resolve()),
        "origem_bytes": origem.stat().st_size,
        "origem_linhas": total_linhas,
        "saida": destino.name,
        "saida_bytes": destino.stat().st_size,
        "saida_linhas": len(aceitas),
        "sha256": sha.hexdigest(),
        "semente": args.seed,
        "somente_camaras": args.somente_camaras,
        "linhas_sorteadas_antes_dos_filtros": len(sorteadas),
        "descartes": dict(descartes),
        "tipo_decisao": dict(Counter(item[2] for item in aceitas)),
        "codigo_tribunal_cnj": dict(Counter(item[3] for item in aceitas).most_common(10)),
    }
    caminho_manifesto = destino.with_suffix(".manifest.json")
    caminho_manifesto.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nAmostra gravada: {destino} ({destino.stat().st_size / 1024 ** 2:,.0f} MB, {len(aceitas):,} linhas)")
    print(f"Tipo de decisao : {manifesto['tipo_decisao']}")
    print(f"Codigo CNJ      : {manifesto['codigo_tribunal_cnj']}")
    print(f"Descartes       : {manifesto['descartes']}")
    print(f"SHA-256         : {manifesto['sha256']}")
    print(f"Manifesto       : {caminho_manifesto}")
    print(f"Tempo total     : {_tempo(time.time() - inicio)}")


if __name__ == "__main__":
    main()
