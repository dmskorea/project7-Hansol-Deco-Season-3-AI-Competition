## Comment
# data pipeline a5
# freeze_layers: 18 layers
# optimizer: adamw_8bit
# doc_str: 사고원인, 재발방지대책 및 향후조치계획
# increase batch size 24 to 32
# 코드 리팩토링
# augmentation 방법 심플하게 변경 (Q + Q&D) 후 Q 쪽을 반반 섞음 -> 어차피 정확히 query와 일치하는 doc이 없을 것으로 예상
# context var 로 인적사고 및 계절/기온 추가
# label 시 target 정보 활용
# 외부자료 활용을 위해 '사고원인', target_var 외 나머지 정보 학습 x
# 외부자료도 학습에 추가하고 context var 은 '없음' 으로 대체
# 온도 augmentation 추가
# 부위(위치) 추가 및 나열 순서 변경
# batchsize 증가 (32 to 128)
# qwen augmentation 자료 추가

SEED = 42
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from numpy import random as np_rnd
import pandas as pd
import torch
from datetime import datetime
import datasets
import torch
from transformers import AutoTokenizer
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import logging
import gc
from utils import *

class CFG:
    prj_path = "/hdd_data2/yjk/dacon-hansol3-dpr/"
    dp_path = "/hdd_data2/yjk/dacon-hansol3/dataset/prepdata/"
    dp_version = "a6"
    ext_version = "e1"
    target_var = "재발방지대책 및 향후조치계획"
    debug = False
    ddp = False
    architecture_name = "dpr-aug_bge-m3_sft_bS9-2_c6"
    model_id = "/hdd_data2/llm_archive/bge-m3/"
    freeze_layers = ["embeddings"] + [f"encoder.layer.{i}" for i in range(18)] + ["pooler"]
    max_length = 1024
    epochs = 2 if debug else 5
    early_stopping_rounds = 5
    eval_steps = 0.5 if debug else 0.1
    train_batch_size = 4 if debug else 128
    valid_batch_size = 4 if debug else 128
    eta = 2e-5
    weight_decay = 1e-2
    reranking_top_k = 5
    replace_str = "없음"

if CFG.ddp:
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP

# create output directory and initialize logger
seed_everything(SEED)
dt_now = datetime.now().strftime('%Y-%m-%d_%H-%M')
architecture_path = os.path.join(CFG.prj_path, f"dpr_architectures/{CFG.architecture_name}_{dt_now}")
createFolder(architecture_path)
logging.basicConfig(filename=os.path.join(architecture_path, "log.txt"), filemode="a", level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
context_str = ["계절/기온", "부위", "인적사고"]
max_len = 300
nan_str = ["", "-", "없음"]
replace_str = "없음"

def group_preprocess(x):
    return " ".join([i for i in x if (i not in nan_str) and (i != replace_str)])

def create_labels(target, corpus):
    model_sts = SentenceTransformer("/hdd_data2/llm_archive/bge-m3/", tokenizer_kwargs={"max_length": 1024})
    labels = []
    target_embed = model_sts.encode(target, batch_size=CFG.valid_batch_size, normalize_embeddings=True)
    corpus_embed = model_sts.encode(corpus, batch_size=CFG.valid_batch_size, normalize_embeddings=True)
    scores = target_embed @ corpus_embed.T
    indices = np.argsort(scores, axis=1)[:, ::-1]
    labels = pd.DataFrame(indices[:, :CFG.reranking_top_k], dtype="str").apply(lambda x: ",".join(x), axis=1)
    del model_sts
    gc.collect()
    torch.cuda.empty_cache()
    return labels.to_list()

def formatting_prompts_func(row: pd.Series, aug=False):
    tmp = []
    for col, val in row.items():
        if col == "인적사고":
            tmp.append(f"사고유형: {val[:max_len]}")
        elif col == "부위":
            tmp.append(f"위치: {val[:max_len]}")
        elif aug and (col == "계절/기온"):
            text = val[:max_len]
            text = text.split("/")
            if len(text) >= 2:
                try:
                    text = text[0] + "/" +str(int(text[-1].rstrip('°C')) + np_rnd.randint(-1, 2)) + "°C"
                except:
                    text = CFG.replace_str
            else:
                text = text[0]
            tmp.append(f"{col}: {text}")
        else:
            tmp.append(f"{col}: {val[:max_len]}")
    return "\n".join(tmp)

def create_evaldata(queries, corpus, labels):
    queries = {k: v for k, v in enumerate(queries)}
    corpus = {k: v for k, v in enumerate(corpus)}
    relevant_docs = {k: set([int(i) for i in v.split(",")]) for k, v in enumerate(labels)}
    return {"corpus": corpus, "queries": queries, "relevant_docs": relevant_docs}

def main():   
    if CFG.ddp:
        # initialize distributed setting
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["LOCAL_RANK"])
        logger.info(f"initialize DDP / world_size={world_size}, rank={rank}")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    # loading data
    df_train = pickleIO(None, os.path.join(CFG.dp_path, CFG.dp_version, "df_full.pkl"), "r")
    df_valid = pickleIO(None, os.path.join(CFG.dp_path, CFG.dp_version, "df_valid.pkl"), "r")
    if CFG.debug:
        df_train = df_train.iloc[:100]
        df_valid = df_valid.iloc[:100]

    # create eval data
    queries = pd.DataFrame({
        **{col: df_valid[col].to_list() for col in context_str},
        "사고원인": df_valid[f"사고원인"].to_list(),
    }).apply(lambda x: formatting_prompts_func(x, aug=True), axis=1).to_list()
    corpus = pd.DataFrame({
        "사고원인": df_train[f"aug_사고원인"].to_list(),
        CFG.target_var: df_train[CFG.target_var].to_list(),
    }).apply(lambda x: formatting_prompts_func(x), axis=1).to_list()
    labels = create_labels(
        df_valid[["사고원인", CFG.target_var]].apply(lambda x: "\n".join(x), axis=1).to_list(),
        df_train[["사고원인", CFG.target_var]].apply(lambda x: "\n".join(x), axis=1).to_list()
    )

    df_train = datasets.Dataset.from_dict({
        "sentence1": pd.DataFrame({
            **{col: df_train[col].to_list() * 2 for col in context_str},
            "사고원인": df_train["aug_사고원인"].to_list() + df_train["사고원인"].to_list(),
        }).apply(lambda x: formatting_prompts_func(x, aug=True), axis=1).to_list(),
        "sentence2": pd.DataFrame({
            "사고원인": df_train["사고원인"].to_list() + df_train["aug_사고원인"].to_list(),
            CFG.target_var: df_train[CFG.target_var].to_list() + df_train[f"aug_{CFG.target_var}"].to_list(),
        }).apply(lambda x: formatting_prompts_func(x), axis=1).to_list(),
    })
    df_valid = datasets.Dataset.from_dict({
        "sentence1": pd.DataFrame({
            **{col: df_valid[col].to_list() for col in context_str},
            "사고원인": df_valid["사고원인"].to_list(),
        }).apply(lambda x: formatting_prompts_func(x, aug=True), axis=1).to_list(),
        "sentence2": pd.DataFrame({
            "사고원인": df_valid["aug_사고원인"].to_list(),
            CFG.target_var: df_valid[CFG.target_var].to_list(),
        }).apply(lambda x: formatting_prompts_func(x), axis=1).to_list(),
    })
    logger.info(f"sentences example:")
    logger.info("Q: " + df_train[0]["sentence1"] + " D: " + df_train[0]["sentence2"] + "\n\n")
    logger.info("Q: " + df_train[1]["sentence1"] + " D: " + df_train[1]["sentence2"])

    # Load a model to train/finetune
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
    model = SentenceTransformer(CFG.model_id, tokenizer_kwargs={"max_length": CFG.max_length})
    for n, p in model[0].auto_model.named_parameters():
        if any(n.startswith(t) if t.startswith("embedding") or t.startswith("pooler") else ".".join(n.split(".")[:3]) == t for t in CFG.freeze_layers):
            p.requires_grad = False
        else:
            logger.info(f"{n} is trainable")
            p.requires_grad = True  

    # create evaluator
    logger.info("=== Evaluation before training ===")
    K = [1, 3, 5]
    ir_evaluator = InformationRetrievalEvaluator(
        **create_evaldata(queries, corpus, labels), batch_size=CFG.valid_batch_size,
        mrr_at_k=[K[-1]],
        ndcg_at_k=[K[-1]],
        accuracy_at_k=K,
        precision_recall_at_k=K
    )
    ir_evaluator(model)

    # specify training arguments
    args = SentenceTransformerTrainingArguments(
        num_train_epochs=CFG.epochs,
        per_device_train_batch_size=CFG.train_batch_size,
        per_device_eval_batch_size=CFG.valid_batch_size,
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        learning_rate=CFG.eta,
        weight_decay=CFG.weight_decay,
        lr_scheduler_type="linear",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_drop_last=True,
        bf16=True if torch.cuda.is_available() else False,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=CFG.eval_steps,
        metric_for_best_model="eval_loss",
        save_strategy="steps",
        save_steps=CFG.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=CFG.eval_steps,
        report_to="none",
        output_dir=os.path.join(architecture_path, "runs"),
        overwrite_output_dir=True,
        seed=SEED,
    )

    # create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=df_train,
        eval_dataset=df_valid,
        loss=MultipleNegativesRankingLoss(model),
        evaluator=ir_evaluator,
    )
    trainer.train()
    model.save_pretrained(os.path.join(architecture_path, "model_best"))
    tokenizer.save_pretrained(os.path.join(architecture_path, "model_best"))

    if CFG.ddp:
        dist.destroy_process_group()

    logger.info("=== Evaluation after training ===")
    ir_evaluator(model)
    
if __name__ == "__main__":
    main()
