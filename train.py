import os
from typing import List, Tuple

import click
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from mlflow import log_metric, log_param, set_experiment, set_tracking_uri, start_run
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoTokenizer,
    BertForSequenceClassification,
    DebertaForSequenceClassification,
    RobertaForSequenceClassification,
    models,
)

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def set_run_name(model_name: str, use_peft: bool) -> str:
    w = "w/" if use_peft else "w/o"
    return f"{model_name: <7} {w: <3} peft"


def set_model(model_name: str, num_labels: int, use_peft: bool):
    if model_name == "deberta":
        model_name = "ku-nlp/deberta-v2-base-japanese"
        model = DebertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
    elif model_name == "roberta":
        model_name = "nlp-waseda/roberta-base-japanese"
        model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
    else:
        model_name = "cl-tohoku/bert-base-japanese-v2"
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="multi_label_classification"
        )

    if use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    return model, model_name


def make_datasets(
    training_file_name: str,
    validation_file_name: str,
    model_name: str,
) -> Tuple[Dataset, Dataset]:
    training_df = pd.read_json(training_file_name, lines=True).loc[
        :, ["text", "labels"]
    ]
    validation_df = pd.read_json(validation_file_name, lines=True).loc[
        :, ["text", "labels"]
    ]

    mlb = MultiLabelBinarizer()
    mlb.fit(training_df["labels"].values)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_dataset = make_dataset(training_df, tokenizer, mlb)
    validation_dataset = make_dataset(validation_df, tokenizer, mlb)

    return training_dataset, validation_dataset


def make_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    mlb: MultiLabelBinarizer,
) -> Dataset:
    df["labels"] = df["labels"].apply(lambda x: mlb.transform([x])[0].astype(float))

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda row: tokenizer(
            row["text"], truncation=True, padding="max_length", max_length=20
        ),
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )

    return dataset


def train(
    run_name: str,
    model: models,
    model_name: str,
    training_dataset: Dataset,
    validation_dataset: Dataset,
    batch_size: int,
    epoch_num: int,
    mlflow_endpoint: str,
) -> None:
    optimizer = AdamW(model.parameters(), lr=1e-5)

    training_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )

    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True
    )

    set_tracking_uri(mlflow_endpoint)
    set_experiment("bert-clf")
    os.getenv("LOGNAME")
    with start_run(run_name=run_name):
        for epoch in range(epoch_num):
            print(epoch)
            _train(model, training_dataloader, optimizer, epoch)
            _validate(model, validation_dataloader, epoch)


def _train(
    model: models,
    dataloader: DataLoader,
    optimizer: AdamW,
    epoch: int,
) -> Tuple[float, float]:
    model.train()

    loss = 0.0
    expected_labels = []
    threshold_to_predicted_labels = {}

    for batch in dataloader:
        optimizer.zero_grad()
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
        )
        _loss = output.loss
        loss += _loss.item()

        _loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        expected_labels.extend(batch["labels"].tolist())

        # predict labels for each threshold
        probablities = torch.sigmoid(output.logits)
        for threshold in THRESHOLDS:
            predicted_labels = torch.where(probablities > threshold, 1.0, 0.0).tolist()
            threshold_to_predicted_labels.setdefault(threshold, []).extend(
                predicted_labels
            )

    log_metric("training / loss", loss, step=epoch)
    for threshold in THRESHOLDS:
        predicted_labels = threshold_to_predicted_labels[threshold]
        _evaluate("training", expected_labels, predicted_labels, threshold, epoch)

    return loss


def _validate(model: models, dataloader: DataLoader, epoch: int) -> Tuple[float, float]:
    model.eval()

    loss = 0.0
    expected_labels = []
    threshold_to_predicted_labels = {}

    for batch in dataloader:
        with torch.no_grad():
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=batch["labels"],
            )
            _loss = output.loss
            loss += _loss.item()

            expected_labels.extend(batch["labels"].tolist())

            # predict labels for each threshold
            probablities = torch.sigmoid(output.logits)
            for threshold in THRESHOLDS:
                predicted_labels = torch.where(
                    probablities > threshold, 1.0, 0.0
                ).tolist()
                threshold_to_predicted_labels.setdefault(threshold, []).extend(
                    predicted_labels
                )

    log_metric("validation / loss", loss, step=epoch)
    for threshold in THRESHOLDS:
        predicted_labels = threshold_to_predicted_labels[threshold]
        _evaluate("validation", expected_labels, predicted_labels, threshold, epoch)


def _evaluate(
    phase: str,
    expected: List,
    predicted: List,
    threshold: float,
    epoch: int,
) -> None:
    expected = np.array(expected)
    predicted = np.array(predicted)
    log_metric(
        f"{phase} / threshold {threshold} / macro accuracy",
        accuracy_score(expected, predicted),
        step=epoch,
    )
    log_metric(
        f"{phase} / threshold {threshold} / macro precision",
        precision_score(expected, predicted, average="macro"),
        step=epoch,
    )
    log_metric(
        f"{phase} / threshold {threshold} / macro recall",
        recall_score(expected, predicted, average="macro"),
        step=epoch,
    )
    log_metric(
        f"{phase} / threshold {threshold} / macro f1",
        f1_score(expected, predicted, average="macro"),
        step=epoch,
    )


@click.command()
@click.option(
    "--model_name",
    type=click.Choice(["bert", "roberta", "deberta"]),
    default="bert",
)
@click.option(
    "--training_file_name",
    type=str,
    default="data/train.jsonl",
)
@click.option(
    "--validation_file_name",
    type=str,
    default="data/val.jsonl",
)
@click.option(
    "--num_labels",
    type=int,
)
@click.option(
    "--epoch_num",
    type=int,
    default=30,
)
@click.option(
    "--batch_size",
    type=int,
    default=256,
)
@click.option(
    "--mlflow_endpoint",
    type=str,
)
@click.option(
    "--use_peft",
    type=bool,
    default=False,
)
def main(
    model_name: str,
    training_file_name: str,
    validation_file_name: str,
    num_labels: int,
    batch_size: int,
    epoch_num: int,
    mlflow_endpoint: str,
    use_peft: bool,
) -> None:

    run_name = set_run_name(model_name, use_peft)
    model_dir_name = f"models/{model_name}"
    model, model_name = set_model(model_name, num_labels, use_peft)
    training_dataset, validation_dataset = make_datasets(
        training_file_name, validation_file_name, model_name
    )
    train(
        run_name,
        model,
        model_name,
        training_dataset,
        validation_dataset,
        batch_size,
        epoch_num,
        mlflow_endpoint,
    )
    model.save_pretrained(model_dir_name)


if __name__ == "__main__":
    main()
