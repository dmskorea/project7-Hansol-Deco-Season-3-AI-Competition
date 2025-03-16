## Comment
# data pipeline a5
# lora r=16, lora_alpha=32, lora_dropout=0.05
# optimizer: adamw_8bit
# using unsloth model
# increase batch size from 24 to 64
# RAG style finetuning
# adjust batchsize to 32
# retrieval 로직 변경
# embedding model bge-m3
# prompt p9 버전 사용
# increase max_length to 3072
# retrieval 로직 변경
# prompt p12 버전 사용 (hit 못 할 경우 대비 프롬프트 수정)
# full prompt tuning
# use RAFT architecture
# Q(aug) + C -> D / Q -> D

SEED = 42
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from numpy import random as np_rnd
import pandas as pd
from sentence_transformers import SentenceTransformer
import datasets
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datetime import datetime
import logging
import gc
from utils import *

class CFG:
    prj_path = "/hdd_data2/yjk/dacon-hansol3-dpr/"
    dp_path = "/hdd_data2/yjk/dacon-hansol3/dataset/prepdata/"
    dp_version = "a5"
    ext_version = "e1"
    debug = False
    target_var = "재발방지대책 및 향후조치계획"
    architecture_name = "sftUS-ragRAFT-extAug-base_llama3.1-8b_bS9-1_p16"  
    model_id = "/hdd_data2/llm_archive/Llama-3.1-8B-Instruct"
    embedding_model_id = "/hdd_data2/yjk/dacon-hansol3-dpr/dpr_architectures/dpr-aug_bge-m3-ext_sft_bS6-8_c6_2025-03-13_19-35/model_best"
    max_length = 2048
    epochs = 2 if debug else 3
    early_stopping_rounds = 5
    eval_steps = 0.5 if debug else 0.1
    train_batch_size = 4 if debug else 32
    valid_batch_size = 4 if debug else 32
    eta = 2e-5
    weight_decay = 1e-2
    retrieval_top_k = 5
    replace_str = "없음"
dt_now = datetime.now().strftime('%Y-%m-%d_%H-%M')
seed_everything(SEED)

# create output directory and initialize logger
architecture_path = os.path.join(CFG.prj_path, f"sft_architectures/{CFG.architecture_name}_{dt_now}")
createFolder(architecture_path)
logging.basicConfig(filename=os.path.join(architecture_path, "log.txt"), filemode="a", level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

rag_prompt_template = {
    "user": """ 
과거 사례에서 제시된 재발방지대책 및 향후조치계획을 참고하여, 이번 사례에 적합한 재발방지대책 및 향후조치계획을 300자 이내로 작성해 주세요.
과거 사례 및 이번 사례의 내용을 종합하여 활용하되, 새로운 정보는 추가하지 마세요.
응답을 콤마로 구분된 일반 텍스트 형식으로 작성하며, markdown 형식을 사용하지 마세요.

## 과거 사례
{contexts}

## 이번 사례
계절/기온: {season}
위치: {location}
사고유형: {type}
사고원인: {query}
재방지대책 및 향후조치계획:
"""
}
qa_prompt_template = {
    "user": """ 
이번 사례에 적합한 재발방지대책 및 향후조치계획을 300자 이내로 작성해 주세요.
응답을 콤마로 구분된 일반 텍스트 형식으로 작성하며, markdown 형식을 사용하지 마세요.

## 이번 사례
계절/기온: {season}
위치: {location}
사고유형: {type}
사고원인: {query}
재방지대책 및 향후조치계획:
"""
}
response_template = "## 이번 사례"
contexts_vars = ["계절/기온", "부위", "인적사고", CFG.target_var]
query_str = ["계절/기온", "부위", "인적사고", "사고원인"]
doc_str = ["사고원인", CFG.target_var]
separator = "\n"
max_len = 300

def formatting_prompts_func(example, prompt_template):
    if "contexts" in example:
        prompt = {
            "contexts": example['contexts'],
            "season": example['계절/기온'],
            "location": example['부위'],
            "type": example['인적사고'],
            "query": example['사고원인'][:max_len],
        }
    else:
        prompt = {
            "season": example['계절/기온'],
            "location": example['부위'],
            "type": example['인적사고'],
            "query": example['사고원인'][:max_len],
        }
    if "system" in prompt_template:
        messages = [
            {"role": "system", "content": prompt_template["system"].strip()},
            {"role": "user", "content": prompt_template["user"].format(**prompt).lstrip()},
            {"role": "assistant", "content": example[CFG.target_var].strip()[:max_len]},
        ]
    else:
        messages = [
            {"role": "user", "content": prompt_template["user"].format(**prompt).lstrip()},
            {"role": "assistant", "content": example[CFG.target_var].strip()[:max_len]},
        ]
    return {"messages": messages}

def print_trainable_parametes(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params : {trainable_params} || all params : {all_param} || trainable% : {100 * trainable_params / all_param}")

def create_contexts(row: pd.Series, aug=False):
    tmp = []
    for col, val in row.items():
        if col == "인적사고":
            tmp.append(f"사고유형: {val[:max_len]}")
        elif col == "부위":
            tmp.append(f"{'위치'}: {val[:max_len]}")
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

def retrieval_augmentation(shuffled_indices, indices, df_corpus, cols, aug):
    data = []
    indices_buckets = np.array_split(shuffled_indices, 20)
    def fn_formatting(contexts):
        return "\n".join([f"사례{i+1}) {context}" for i, context in enumerate(contexts)])
    # 15%: top5
    for i in np.concatenate([indices_buckets.pop(0) for _ in range(3)]):
        idx = indices[i, 1:6].tolist()
        contexts = df_corpus[cols].iloc[idx].apply(
            lambda x: create_contexts(pd.Series({k.lstrip("aug_"): v for k, v in x.items()}), aug), axis=1
        ).to_list()
        contexts = fn_formatting(contexts)
        data.append(contexts)
    # 50%: top6 ~ top10% 구간 랜덤샘플링
    for i in np.concatenate([indices_buckets.pop(0) for _ in range(10)]):
        idx = [indices[i, 1]] + np_rnd.choice(indices[i, 6:(indices.shape[1] // 10)], CFG.retrieval_top_k - 1, replace=CFG.debug).tolist()
        contexts = df_corpus[cols].iloc[idx].apply(
            lambda x: create_contexts(pd.Series({k.lstrip("aug_"): v for k, v in x.items()}), aug), axis=1
        ).to_list()
        contexts = fn_formatting(contexts)
        data.append(contexts)
    # 30%: top10% ~ top50% 구간 랜덤샘플링
    for i in np.concatenate([indices_buckets.pop(0) for _ in range(6)]):
        idx = [indices[i, 1]] + np_rnd.choice(indices[i, (indices.shape[1] // 10):(indices.shape[1] // 2)], CFG.retrieval_top_k - 1, replace=CFG.debug).tolist()
        contexts = df_corpus[cols].iloc[idx].apply(
            lambda x: create_contexts(pd.Series({k.lstrip("aug_"): v for k, v in x.items()}), aug), axis=1
        ).to_list()
        contexts = fn_formatting(contexts)
        data.append(contexts)
    # 5%: 하위 50% 구간 랜덤샘플링
    for i in np.concatenate([indices_buckets.pop(0) for _ in range(1)]):
        idx = [indices[i, 1]] + np_rnd.choice(indices[i, (indices.shape[1] // 2):], CFG.retrieval_top_k - 1, replace=CFG.debug).tolist()
        contexts = df_corpus[cols].iloc[idx].apply(
            lambda x: create_contexts(pd.Series({k.lstrip("aug_"): v for k, v in x.items()}), aug), axis=1
        ).to_list()
        contexts = fn_formatting(contexts)
        data.append(contexts)
    return data

def create_retrieval(df_train, df_valid, aug=False):
    df_corpus = pd.concat([df_train, df_valid], axis=0)
    model_sts = SentenceTransformer(CFG.embedding_model_id, tokenizer_kwargs={"max_length": 1024})
    corpus_embed = model_sts.encode(df_corpus[doc_str].apply(create_contexts, axis=1).to_list(), batch_size=CFG.valid_batch_size, normalize_embeddings=True)
    train_embed = model_sts.encode(df_train[query_str].apply(create_contexts, axis=1).to_list(), batch_size=CFG.valid_batch_size, normalize_embeddings=True)
    valid_embed = model_sts.encode(df_valid[query_str].apply(create_contexts, axis=1).to_list(), batch_size=CFG.valid_batch_size, normalize_embeddings=True)
    # for train data
    shuffled_indices = np_rnd.permutation(len(train_embed))
    scores = ((train_embed @ corpus_embed.T) + 1) / 2
    indices = np.argsort(scores, axis=1)[:, ::-1]
    data = retrieval_augmentation(
        shuffled_indices, indices, df_corpus,
        [f"aug_{col}" if col in ["사고원인", CFG.target_var] else col for col in contexts_vars] if aug else contexts_vars, aug
    )
    if aug:
        df_train["aug_contexts"] = ""
        df_train["aug_contexts"].iloc[shuffled_indices] = data
    else:
        df_train["contexts"] = ""
        df_train["contexts"].iloc[shuffled_indices] = data
    # for valid data
    shuffled_indices = np_rnd.permutation(len(valid_embed))
    scores = ((valid_embed @ corpus_embed.T) + 1) / 2
    indices = np.argsort(scores, axis=1)[:, ::-1]
    data = retrieval_augmentation(
        shuffled_indices, indices, df_corpus,
        [f"aug_{col}" if col in ["사고원인", CFG.target_var] else col for col in contexts_vars] if aug else contexts_vars, aug
    )
    if aug:
        df_valid["aug_contexts"] = ""
        df_valid["aug_contexts"].iloc[shuffled_indices] = data
    else:
        df_valid["contexts"] = ""
        df_valid["contexts"].iloc[shuffled_indices] = data
    del model_sts
    gc.collect()
    torch.cuda.empty_cache()
    return df_train, df_valid

def main():
    # create dataset
    df_train = pickleIO(None, os.path.join(CFG.dp_path, CFG.dp_version, "df_full.pkl"), "r")
    df_valid = pickleIO(None, os.path.join(CFG.dp_path, CFG.dp_version, "df_valid.pkl"), "r")
    df_ext = pd.read_csv(f"/hdd_data2/yjk/dacon-hansol3/dataset/extdata/{CFG.ext_version}/external_docs_qa.csv")
    df_ext[["aug_사고원인", f"aug_{CFG.target_var}"]] = pd.read_csv(f"/hdd_data2/yjk/dacon-hansol3/dataset/extdata/{CFG.ext_version}/external_docs_qa_aug.csv")[["사고원인", CFG.target_var]]
    df_train = pd.concat([df_train, df_ext], axis=0).fillna(CFG.replace_str).reset_index(drop=True)
    if CFG.debug:
        df_train = df_train.iloc[:100]
        df_valid = df_valid.iloc[:100]
    df_train["부위"] = df_train["부위_level0"] + " " + df_train["부위_level1"]
    df_valid["부위"] = df_valid["부위_level0"] + " " + df_valid["부위_level1"]
    # create retrieval data
    df_train, df_valid = create_retrieval(df_train, df_valid)
    df_train, df_valid = create_retrieval(df_train, df_valid, aug=True)
    df_train = df_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # create dataset
    df_train_rag = datasets.Dataset.from_dict({
        "contexts": df_train["contexts"].to_list() + df_train["aug_contexts"].to_list(),
        "계절/기온": df_train["계절/기온"].to_list() * 2,
        "부위": df_train["부위"].to_list() * 2,
        "인적사고": df_train["인적사고"].to_list() * 2,
        "사고원인": df_train["aug_사고원인"].to_list() + df_train["사고원인"].to_list(),
        CFG.target_var: df_train[CFG.target_var].to_list() + df_train[CFG.target_var].to_list(),
    }).train_test_split(test_size=0.5, seed=SEED)["test"]
    df_train_qa = datasets.Dataset.from_dict({
        "사고원인": df_train["사고원인"].to_list(),
        "계절/기온": df_train["계절/기온"].to_list(),
        "부위": df_train["부위"].to_list(),
        "인적사고": df_train["인적사고"].to_list(),
        CFG.target_var: df_train[f"aug_{CFG.target_var}"].to_list(),
    }).train_test_split(test_size=0.5, seed=SEED)["test"]
    df_valid = datasets.Dataset.from_dict({
        "contexts": df_valid["contexts"].to_list(),
        "계절/기온": df_valid["계절/기온"].to_list(),
        "부위": df_valid["부위"].to_list(),
        "인적사고": df_valid["인적사고"].to_list(),
        "사고원인": df_valid["aug_사고원인"].to_list(), 
        CFG.target_var: df_valid[CFG.target_var].to_list(),
    })
    df_train = datasets.concatenate_datasets([
        df_train_rag.map(lambda x: formatting_prompts_func(x, rag_prompt_template), remove_columns=df_train_rag.column_names),
        df_train_qa.map(lambda x: formatting_prompts_func(x, qa_prompt_template), remove_columns=df_train_qa.column_names),
    ])
    df_valid = df_valid.map(lambda x: formatting_prompts_func(x, rag_prompt_template), remove_columns=df_valid.column_names)
    logger.info(f"message example:")
    logger.info(df_train[0]["messages"][0]["content"] + " " + df_train[0]["messages"][1]["content"] + "\n\n")
    logger.info(df_train[1]["messages"][0]["content"] + " " + df_train[1]["messages"][1]["content"])

    # loading model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG.model_id,
        max_seq_length=CFG.max_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        gpu_memory_utilization=0.95,
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        max_seq_length=CFG.max_length,
        random_state=SEED,
    )
    print_trainable_parametes(model)

    # create collator
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # create sft config
    sft_config = SFTConfig(
        max_seq_length=CFG.max_length,
        gradient_checkpointing=True,
        num_train_epochs=CFG.epochs,
        per_device_train_batch_size=CFG.train_batch_size,
        per_device_eval_batch_size=CFG.valid_batch_size,
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        learning_rate=CFG.eta,
        weight_decay=CFG.weight_decay,
        lr_scheduler_type="linear",
        dataloader_drop_last=True,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
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

    # create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=df_train,
        eval_dataset=df_valid,
        data_collator=collator,
        args=sft_config,
    )

    # train
    trainer.train()

    # save model
    model.save_pretrained_merged(os.path.join(architecture_path, "model_best"), tokenizer, save_method="merged_16bit")
    model.save_pretrained(os.path.join(architecture_path, "adaptor_best"))

if __name__ == "__main__":
    main()
