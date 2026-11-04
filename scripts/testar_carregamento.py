"""
Compara as duas formas de carregar o Qwen3-Embedding-4B no sentence-transformers instalado:

  A. "manual": models.Transformer + Pooling(lasttoken) + Normalize, como o model_builder fazia ate
     a correcao. No diagnostico de 03/11/2026 ela deu MRR@10 0,04 na validacao do passo 0.
  B. "hub": SentenceTransformer(model_id), como a mineracao, a Etapa 5 e o model_builder corrigido.

Para cada forma mostra o lado do padding, os modulos, os tokens do fim de um texto e o MRR@10
numa validacao pequena. Mostra tambem o cosseno entre as saidas das duas formas para os mesmos
textos (1,0 = mesmos vetores) e confere o construir_modelo_embedding do treino.

Uso (uma GPU, ~5 min):
    python scripts/testar_carregamento.py --dados data/processed_tjrs --processos 100
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
from sentence_transformers import SentenceTransformer, models

from src.evaluation.ir_evaluator import criar_ir_evaluator
from src.training.model_builder import construir_modelo_embedding

INSTRUCAO = (
    "Instruct: Dado um trecho de um documento jurídico brasileiro, recupere o trecho mais relevante "
    "semanticamente.\nQuery: "
)
TEXTOS = [
    INSTRUCAO + "APELAÇÃO CÍVEL. RESPONSABILIDADE CIVIL. DANOS MORAIS. INSCRIÇÃO INDEVIDA EM CADASTRO DE INADIMPLENTES.",
    "Trata-se de recurso de apelação interposto contra sentença que julgou procedente a ação de indenização.",
    "A inscrição indevida do nome do consumidor em órgão de proteção ao crédito gera dano moral in re ipsa.",
    "Ante o exposto, voto por negar provimento ao recurso.",
]


def montar_manual(model_id: str, dtype: torch.dtype, max_seq: int) -> SentenceTransformer:
    transformer = models.Transformer(
        model_name_or_path=model_id,
        max_seq_length=max_seq,
        tokenizer_args={"padding_side": "left"},
        model_args={"attn_implementation": "sdpa", "torch_dtype": dtype, "trust_remote_code": True},
    )
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="lasttoken")
    return SentenceTransformer(modules=[transformer, pooling, models.Normalize()], device="cuda:0")


def montar_hub(model_id: str, dtype: torch.dtype, max_seq: int) -> SentenceTransformer:
    modelo = SentenceTransformer(
        model_id,
        device="cuda:0",
        trust_remote_code=True,
        model_kwargs={"attn_implementation": "sdpa", "torch_dtype": dtype},
        tokenizer_kwargs={"padding_side": "left"},
    )
    modelo.max_seq_length = max_seq
    return modelo


def descrever(nome: str, modelo: SentenceTransformer) -> None:
    tok = getattr(modelo, "tokenizer", None)
    print(f"\n=== {nome} ===")
    print(f"Modulos : {' -> '.join(type(m).__name__ for m in modelo)}")
    print(f"Padding : {getattr(tok, 'padding_side', '?')} | max_seq_length: {modelo.max_seq_length}")
    for i, m in enumerate(modelo):
        cfg = getattr(m, "get_config_dict", None)
        if cfg and i > 0:
            print(f"Modulo {i}: {cfg()}")
    if tok is not None:
        lote = tok(TEXTOS[:2], padding=True, return_tensors="pt")
        for linha_ids, linha_mask in zip(lote["input_ids"], lote["attention_mask"]):
            print(f"  ultimos tokens: {tok.convert_ids_to_tokens(linha_ids[-3:].tolist())} | "
                  f"mascara inicio/fim: {linha_mask[:2].tolist()} / {linha_mask[-2:].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--dados", default="data/processed_tjrs", help="Pasta com val_processos.jsonl e val_corpus_chunks.jsonl")
    parser.add_argument("--processos", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=768)
    args = parser.parse_args()

    dados = Path(args.dados)
    if not dados.is_absolute():
        dados = BASE_DIR / dados
    avaliador = criar_ir_evaluator(
        caminho_processos=dados / "val_processos.jsonl",
        caminho_chunks=dados / "val_corpus_chunks.jsonl",
        nome="teste_carregamento",
        query_instruction=INSTRUCAO,
        batch_size=64,
        show_progress_bar=False,
        limite_processos=args.processos,
    )

    dtype = torch.bfloat16
    resultados: dict[str, float] = {}
    vetores: dict[str, torch.Tensor] = {}
    construtores = {
        "A_manual": lambda: montar_manual(args.model_id, dtype, args.max_seq_length),
        "B_hub": lambda: montar_hub(args.model_id, dtype, args.max_seq_length),
        "C_model_builder_do_treino": lambda: construir_modelo_embedding(
            model_id=args.model_id, max_seq_length=args.max_seq_length, device="cuda:0", precisao="bf16"
        ),
    }
    for nome, construir in construtores.items():
        modelo = construir()
        modelo.eval()
        descrever(nome, modelo)
        with torch.no_grad():
            vetores[nome] = modelo.encode(TEXTOS, convert_to_tensor=True, batch_size=4).float().cpu()
            metricas = avaliador(modelo)
        mrr = next(v for k, v in metricas.items() if k.endswith("mrr@10"))
        resultados[nome] = mrr
        print(f"MRR@10 em {args.processos} processos de validacao: {mrr:.4f}")
        del modelo
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Resumo ===")
    for nome, mrr in resultados.items():
        print(f"{nome:28s} MRR@10 = {mrr:.4f}")
    ref = vetores["B_hub"]
    for nome, vet in vetores.items():
        if nome != "B_hub":
            cos = torch.nn.functional.cosine_similarity(vet, ref).tolist()
            print(f"Cosseno {nome} x B_hub por texto: {[round(c, 4) for c in cos]}")
    ok = abs(resultados["C_model_builder_do_treino"] - resultados["B_hub"]) < 0.02
    print("\nVeredito: " + ("o modelo do treino agora e igual ao do Hub. Pode rodar o diagnostico."
                            if ok else "ATENCAO: o modelo do treino ainda difere do Hub. Nao rode o treino; mande esta saida."))


if __name__ == "__main__":
    main()
