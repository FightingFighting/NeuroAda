import os
import torch
import argparse
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
    TrainingArguments
)
import transformers
from transformers.trainer_utils import EvalPrediction, has_length, denumpify_detensorize
import wandb
import evaluate
import datetime
import json
import math
import copy
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import Dataset

from task_config import task_config
from dataset import LoReftGLUEDataset, LoReftSupervisedDataset
from compute_metrics import compute_metrics_percell
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import Trainer
from transformers.utils import logging

logger = logging.get_logger(__name__)
import sys


from utils.utils import *
from utils.select_para import *
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import uuid

subtask2metric = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson"
}


def make_dataloader(dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)

class TrainerForCausalLM(Trainer):

    def __init__(self, *args, peft_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.peft_type=peft_type



    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        loss = super().training_step(model, inputs)

        if self.peft_type == "perCell_mag" or self.peft_type == "perCell_gra":
            with torch.no_grad():
                mask_gradient(model, model.mask)


        return loss








class TrainerForSequenceClassification(Trainer):
    def __init__(self, *args, peft_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.peft_type=peft_type

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False
    ):

        cf_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        # classification loss on counterfactual labels
        logits = cf_outputs.logits
        labels = inputs["labels"]

        if self.model.config.problem_type is None:
            if self.model.num_labels == 1:
                problem_type = "regression"
            elif self.model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
        else:
            problem_type = self.model.config.problem_type

        if problem_type == "regression":
            loss_fct = MSELoss()
            if self.model.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze().to(torch.bfloat16))
            else:
                loss = loss_fct(logits, labels.to(torch.bfloat16))
        elif problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        elif problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        # return
        return (loss, cf_outputs) if return_outputs else loss

    def evaluate(self, ignore_keys):
        # ensure everything is in eval mode
        self.model.eval()

        batch_size = self.args.eval_batch_size
        data_collator = self.data_collator
        eval_dataset = self.eval_dataset

        dataloader = make_dataloader(eval_dataset, batch_size, data_collator, shuffle=False)

        logger.info(f"***** Running In-Training Evaluation *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        eval_iterator = tqdm(dataloader, position=0, leave=True)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for step, inputs in enumerate(eval_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.model.device)

                # [layers, batch_size, positions]
                cf_outputs = self.model(
                    **{"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                )

                all_preds += [cf_outputs.logits]
                all_labels += [inputs["labels"]]
        all_preds = torch.cat(all_preds, dim=0).cpu().to(torch.float32)
        all_labels = torch.cat(all_labels, dim=0).cpu().to(torch.float32)
        metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        metrics = denumpify_detensorize(metrics)

        metric_key_prefix = "eval"
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def save_model(self, output_dir, _internal_call=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        state_dict_tunable = {}
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                state_dict_tunable[name] = para

        torch.save(state_dict_tunable, os.path.join(output_dir, "state_dict_tunable.pt"))

    def _load_best_model(self):
        logger.warning(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        tunable_para_state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, "state_dict_tunable.pt"),
                                             map_location=torch.device("cuda"))
        self.model.load_state_dict(tunable_para_state_dict, strict=False)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        if self.peft_type == "perCell_mag" or self.peft_type == "perCell_gra":
            with torch.no_grad():
                mask_gradient(model, model.mask)


        return loss


class SupervisedDataset(LoReftSupervisedDataset):
    def __init__(
            self, task: str, data_path: str,
            tokenizer: transformers.PreTrainedTokenizer,
            data_split="train", dataset=None, seed=42, max_n_example=None,
            **kwargs,
    ):
        super().__init__(task.replace("commonsense_15k", "commonsense"), data_path,
                         tokenizer,
                         data_split, dataset, seed, max_n_example,
                         **kwargs, )

        self.result_temp = []
        for i, data_item in enumerate(tqdm(self.result)):
            data_item.pop('intervention_locations')
            if data_split == "train":
                data_item.pop('id')
            self.result_temp.append(data_item)
        self.result = self.result_temp


class GLUEDataset(LoReftGLUEDataset):
    def __init__(
            self, task: str, data_path: str,
            tokenizer: transformers.PreTrainedTokenizer,
            data_split="train", dataset=None, seed=42, max_n_example=None,
            **kwargs,
    ):
        super().__init__(task, data_path,
                         tokenizer,
                         data_split, dataset, seed, max_n_example,
                         **kwargs, )

        self.result_temp = []
        for i, data_item in enumerate(tqdm(self.result)):
            data_item.pop('intervention_locations')
            if data_split == "train":
                data_item.pop('id')
            self.result_temp.append(data_item)
        self.result = self.result_temp


try:
    # This library is our indicator that the required installs
    # need to be done.
    import peft

    is_peft_available = True
except ModuleNotFoundError:
    is_peft_available = False

device = "cuda" if torch.cuda.is_available() else "cpu"
classification_tasks = {"glue"}
residual_stream_component_mapping = {
    "robertaformaskedlm": "roberta.encoder.layer[%s].output"
}
dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8": "float8",
}


def build_dataset(model_name, max_length, task, train_dataset, eval_dataset, data_dir, max_n_train_example, seed,
                  test_split, share_weights, max_n_eval_example, train_dataset_str):

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name if model_name!="FacebookAI/roberta-large" else "roberta-large",
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False

    # load dataset splits
    assert task in task_config, f"Unrecognized task: {task}"
    train_datasets = task_config[task]["train_datasets"] if train_dataset is None else [train_dataset]
    if task == "glue":
        eval_datasets = [train_dataset]
    else:
        eval_datasets = task_config[task]["eval_datasets"] if eval_dataset is None else [eval_dataset]

    perCellDataset = GLUEDataset if task == "glue" else SupervisedDataset

    train_dataset = perCellDataset(
        task, train_datasets[0] if task == "glue" or task == "ultrafeedback_pair" \
            else (os.path.join(data_dir, train_datasets[0]) if data_dir is not None else train_datasets[0]),
        tokenizer, data_split="train", seed=seed, max_n_example=max_n_train_example,
        **{"num_interventions": 0, "position": 'f7+l7',
           "share_weights": share_weights, "test_split": test_split}
    )
    trigger_tokens = train_dataset.trigger_tokens
    num_labels = train_dataset.num_labels

    all_eval_datasets = {}
    for eval_dataset in eval_datasets:
        test_splits = test_split.split(";")
        all_eval_datasets[eval_dataset] = {}
        for split in test_splits:
            raw_eval = perCellDataset(
                task, eval_dataset if task == "glue" else os.path.join(data_dir, eval_dataset),
                tokenizer, data_split=split, seed=seed, max_n_example=max_n_eval_example,
                **{"num_interventions": 0, "position": 'f7+l7',
                   "share_weights": share_weights}
            )
            all_eval_datasets[eval_dataset][split] = [raw_eval, raw_eval.raw_dataset]
    eval_datasets = all_eval_datasets

    in_train_eval_datasets, in_training_compute_metrics = None, None
    if task == "glue":
        # we repartition the eval_datatsets into [1] 50% validation + [2] 50% test
        # we select the best model on [1] during training
        # we test the selected model on [2] to ensure fairness
        to_split_eval_datasets = eval_datasets[train_dataset_str][test_split][0]
        if len(to_split_eval_datasets) > 5000:
            in_train_n_eval_sample = 1000
        else:
            in_train_n_eval_sample = len(to_split_eval_datasets) // 2

        new_splits = torch.utils.data.random_split(
            to_split_eval_datasets, [len(to_split_eval_datasets) - in_train_n_eval_sample, in_train_n_eval_sample]
        )

        in_test_eval_datasets, in_train_eval_datasets = new_splits[0], new_splits[1]
        eval_datasets[train_dataset_str][test_split][0] = in_test_eval_datasets
        print("GLUE validation split (in training): ", len(in_train_eval_datasets))
        print("GLUE validation split (testing): ", len(eval_datasets[train_dataset_str][test_split][0]))

        is_regression = train_dataset_str == "stsb"
        metric = evaluate.load("glue", train_dataset_str, experiment_id=str(uuid.uuid4()))

        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def in_training_compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result

    return need_resize, num_labels, tokenizer, in_train_eval_datasets, in_training_compute_metrics, train_dataset, eval_datasets, trigger_tokens


def build_model(task, model, num_labels, train_dataset_str, dtype, need_resize, tokenizer):
    # load model based on task type.
    if task in classification_tasks:
        config = AutoConfig.from_pretrained(
            model, num_labels=num_labels,
            finetuning_task=train_dataset_str,
            load_in_8bit=True if dtype == "float8" else False,
            device_map=device
        )
        # full precision loading since usually for small models

        model = AutoModelForSequenceClassification.from_pretrained(
            model,
            config=config,  # just providing the label
            torch_dtype=dtype if dtype != "float8" else None,
            load_in_8bit=True if dtype == "float8" else False,
            device_map=device
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype if dtype != "float8" else None,  # save memory
            load_in_8bit=True if dtype == "float8" else False,
            device_map=device
        )

        config = model.config

    if need_resize:
        model.resize_token_embeddings(len(tokenizer))

    return model, config


def peft_setting(peft_config, peft_type, model, output_dir, tuneBias, dtype, train_dataset, train_batch_size, data_collator, ig_layer_index=-1):
    if peft_type == "lora":
        pass
    elif "perCell" in peft_type:
        # mask or position
        if peft_type=="perCell_gra_add":
            train_loader = make_dataloader(train_dataset, train_batch_size, data_collator, shuffle=False)
        else:
            train_loader = None
        mask, positions = generate_MaskOrPosition(model, peft_type, train_loader, peft_config["times_num"],
                                                  peft_config["target_modules"], peft_config["selectallhead"],
                                                  tuneBias=tuneBias, ig_layer_index=ig_layer_index)
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "positions.npy"), positions)

        if positions != None:
            _, _, indices_all = update_model(model, positions, time_num=peft_config["times_num"], peft_type=peft_type,
                                             dtype=dtype)
            if indices_all != None:
                np.save(os.path.join(output_dir, "indices.npy"), indices_all)

        if mask !=None:
            model.mask=mask

    print(
        "============================================= model require_grad =================================================")
    for name, para in model.named_parameters():
        print(name, "require_grad:", para.requires_grad)
    print("==============================================================================================", flush=True)
    return model

def evaluate_model(output_dir, file_name, eval_datasets, model, task, tokenizer, trigger_tokens, wandb_entity,
                   eval_batch_size, data_collator, greedy_decoding, temperature, top_p, top_k):
    eval_results = {}
    for dataset_name in eval_datasets:
        # split evalset into chunks
        for split, (eval_dataset, data_items) in eval_datasets[dataset_name].items():

            generations, stats = compute_metrics_percell(
                task, dataset_name, model, tokenizer, eval_dataset, data_items,
                trigger_tokens, wandb_entity, eval_batch_size,
                data_collator if task in classification_tasks else None,
                split, greedy_decoding, temperature, top_p, top_k
            )

            # log
            eval_results.update(stats)
            if wandb.run:
                wandb.log(stats)
            generations = stats if generations is None else generations
            result_json_file_name = f"{output_dir}/{dataset_name}_{split}_outputs_{file_name}.json"
            with open(result_json_file_name, 'w') as json_file:
                json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{output_dir}/eval_results_{file_name}.json"
    with open(result_json_file_name, 'w') as json_file:
        json.dump(eval_results, json_file, indent=4)

    print(f"Training results can be found in {output_dir}")


def finetune(
        model: str,
        epochs: int,
        seed: int,
        max_n_train_example: int,
        max_n_eval_example: int,
        batch_size: int,
        task: str,
        lr: float,
        schedule: str,
        data_dir: str,
        train_dataset: str,
        eval_dataset: str,
        eval_batch_size: int,
        warmup_ratio: float,
        weight_decay: float,
        test_split: str,
        max_length: int,
        dtype: str,
        logging_steps: int,
        share_weights: bool,
        greedy_decoding: bool,
        temperature: float,
        top_p: float,
        top_k: float,
        peft_type: str,
        times_num: int,
        target_modules: list,
        selectallhead: bool,
        wandb_project: str,
        wandb_entity: str,
        wandb_watch: str,
        micro_batch_size: int,
        tuneBias: bool,
        ig_layer_index: int,
        args,
):
    """
    Generic Representation Finetuning.
    """
    # everything is guarded by a single seed
    set_seed(seed)

    if task == "glue":
        metric_for_best_model = subtask2metric[train_dataset]

    assert task in {
        "commonsense", "math", "alpaca", "instruct", "ultrafeedback", "glue", "gsm8k",
        "ultrafeedback_pair", "commonsense_15k"
    }

    dtype = dtype_mapping[dtype]

    model_name = model
    train_dataset_str = train_dataset

    if task in classification_tasks:
        output_dir = os.path.join("output", f"{model_name.replace('/', '-')}", f"nlu", f"{task}", f"{peft_type}",
                                  f"{wandb_entity}")
    else:
        output_dir = os.path.join("output", f"{model_name.replace('/', '-')}", f"nlg", f"{task}", f"{peft_type}",
                                  f"{wandb_entity}")

    # config peft
    if peft_type == "lora":
        pass
    if "perCell" in peft_type:
        peft_config = {}
        peft_config["times_num"] = times_num
        peft_config["target_modules"] = target_modules
        peft_config["selectallhead"] = selectallhead
    if peft_type=="Full":
        peft_config={}

    # build dataset
    need_resize, num_labels, tokenizer, in_train_eval_datasets, in_training_compute_metrics, train_dataset, eval_datasets, trigger_tokens = build_dataset(
        model_name, max_length, task, train_dataset, eval_dataset, data_dir, max_n_train_example, seed,
        test_split, share_weights, max_n_eval_example, train_dataset_str)

    # build model
    model, config = build_model(task, model, num_labels, train_dataset_str, dtype, need_resize, tokenizer)

    # if wandb.run is not None:
    #     wandb.watch(model, log="all", log_freq=100)

    # data_collator
    if task in classification_tasks:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest"
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding="longest"
        )

    if batch_size < micro_batch_size:
        gradient_accumulation_steps = 1
        micro_batch_size = batch_size
    else:
        gradient_accumulation_steps = batch_size // micro_batch_size

    model = peft_setting(peft_config, peft_type, model, output_dir, tuneBias, dtype, copy.deepcopy(train_dataset), micro_batch_size, copy.deepcopy(data_collator), ig_layer_index)

    model.train()


    # # training args
    training_args = TrainingArguments(
        bf16=True if dtype == "bfloat16" else None,
        output_dir=output_dir,
        run_name=wandb_entity,
        num_train_epochs=epochs,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch" if task == "glue" else "no",
        save_strategy="epoch" if task == "glue" else "no",
        metric_for_best_model=metric_for_best_model if task == "glue" else None,
        load_best_model_at_end=True if task == "glue" else False,
        logging_strategy="steps",
        save_total_limit=1,  # for GLUE, it will save 2 at max.
        logging_steps=logging_steps,
        lr_scheduler_type=schedule,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        weight_decay=weight_decay,
        report_to="wandb" if wandb.run else "none",
        use_cpu=False if device == "cuda" else True,
        seed=seed,
        # until HF supports ReFT, this remains False! :)
        remove_unused_columns=False
    )

    # make trainer
    trainer_class = TrainerForSequenceClassification if task in classification_tasks else TrainerForCausalLM
    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=in_train_eval_datasets if task == "glue" else None,
        data_collator=data_collator,
        compute_metrics=in_training_compute_metrics if task == "glue" else None,
        peft_type=peft_type
    )


    if "perCell" in peft_type:
        old_state_dict = model.state_dict

        def filter_state_dict(self):
            return {name: para for name, para in old_state_dict().items() if "new" in name}

        model.state_dict = filter_state_dict.__get__(model, type(model))
    else:
        pass

    trainer.train()

    if wandb.run is not None and task == "glue":
        wandb.log({f"eval/best_{metric_for_best_model}": trainer.state.best_metric})

    # dump config
    args_dict = vars(args)
    json_file_name = f"{output_dir}/args.json"
    with open(json_file_name, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)


    if "perCell" not in args.peft_type:
        model.save_pretrained(os.path.join(output_dir, f"best_model"))

    if "perCell" in args.peft_type:
        state_dict_tunable = {}
        for name, para in model.named_parameters():
            if para.requires_grad:
                state_dict_tunable[name] = para

        os.makedirs(os.path.join(output_dir, "tunable_para"), exist_ok=True)
        torch.save(state_dict_tunable, os.path.join(output_dir, "tunable_para", "state_dict_tunable.pt"))

    model.eval()


    tunable_para_state_dict = torch.load(os.path.join(output_dir, "tunable_para", "state_dict_tunable.pt"),map_location=torch.device("cuda"))
    model.load_state_dict(tunable_para_state_dict, strict=False)
    merge_model(model, tunable_para_state_dict.keys(), peft_type)
    model.eval()
    evaluate_model(output_dir, "final", eval_datasets, model, task, tokenizer, trigger_tokens, wandb_entity,
                   eval_batch_size, data_collator, greedy_decoding, temperature, top_p, top_k)


def main():
    parser = argparse.ArgumentParser(description="A simple script that takes different arguments.")

    parser.add_argument('-task', '--task', type=str, default=None)
    parser.add_argument('-data_dir', '--data_dir', type=str, default="./datasets")
    parser.add_argument('-train_dataset', '--train_dataset', type=str, default=None)
    parser.add_argument('-eval_dataset', '--eval_dataset', type=str, default=None)
    parser.add_argument('-model', '--model', type=str, help='yahma/llama-7b-hf', default='yahma/llama-7b-hf')
    parser.add_argument('-seed', '--seed', type=int, help='42', default=42)
    parser.add_argument('-e', '--epochs', type=int, help='1', default=1)
    parser.add_argument('-max_n_train_example', '--max_n_train_example', type=int, default=None)
    parser.add_argument('-max_n_eval_example', '--max_n_eval_example', type=int, default=None)

    parser.add_argument('-batch_size', '--batch_size', type=int, default=4)
    parser.add_argument('-eval_batch_size', '--eval_batch_size', type=int, default=4)
    parser.add_argument('-lr', '--lr', type=float, default=5e-3)
    parser.add_argument('-schedule', '--schedule', type=str, default='linear')
    parser.add_argument('-wu', '--warmup_ratio', type=float, default=0.00)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.00)
    parser.add_argument('-test_split', '--test_split', type=str, default="validation")
    parser.add_argument('-max_length', '--max_length', type=int, help=512, default=512)
    parser.add_argument('-dtype', '--dtype', type=str, default="bfloat16" if device == "cuda" else "float32")
    parser.add_argument('-logging_steps', '--logging_steps', type=int, help=1, default=1)
    parser.add_argument('-sw', '--share_weights', action='store_true')
    parser.add_argument('-gd', '--greedy_decoding', action='store_true')

    # decoding params
    parser.add_argument('-t', '--temperature', type=float, default=None)
    parser.add_argument('-top_p', '--top_p', type=float, default=None)
    parser.add_argument('-top_k', '--top_k', type=float, default=None)

    # peft
    parser.add_argument("--wandb_project", type=str, default="", help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default="", help="Name of Weights & Biases Entity.")
    parser.add_argument("--wandb_watch", type=str, default="")
    parser.add_argument('--times_num', type=int, default=1)
    parser.add_argument('--ig_layer_index', type=int, default=-1)
    parser.add_argument("--target_modules", nargs='+', default=None)
    parser.add_argument("--selectallhead", action='store_true', default=False)
    parser.add_argument('--peft_type', type=str, default='')
    parser.add_argument('--micro_batch_size', type=int, default=0)
    parser.add_argument("--tuneBias", action='store_true', default=False)

    args = parser.parse_args()

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_entity, config=args)
        args.wandb_entity = args.wandb_entity.replace("/", "-")

    print("++++++++++++++++++++++++args+++++++++++++++++++++++++")
    print(args, flush=True)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++")

    finetune(**vars(args), args=args)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()