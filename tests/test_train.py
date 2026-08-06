import json
from unittest.mock import MagicMock

import pytest

from swsearch import train


def _write_triplets_file(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(i):
    return {"anchor": f"a{i}", "positive": f"p{i}", "negative": f"n{i}", "source": "s", "url": ""}


def test_load_triplet_dataset_streams_anchor_positive_negative_only(tmp_path):
    _write_triplets_file(
        tmp_path / "wiki_0_triplets.jsonl",
        [
            {"anchor": "a1", "positive": "p1", "negative": "n1", "source": "Article A", "url": "http://x"},
            {"anchor": "a2", "positive": "p2", "negative": "n2", "source": "Article B", "url": "http://y"},
        ],
    )

    train_dataset, eval_dataset = train.load_triplet_dataset(str(tmp_path))

    rows = list(train_dataset)
    assert len(rows) == 2
    assert set(rows[0].keys()) == {"anchor", "positive", "negative"}  # source/url dropped
    assert rows[0] == {"anchor": "a1", "positive": "p1", "negative": "n1"}
    assert eval_dataset is None  # only one file -- nothing left to hold out


def test_load_triplet_dataset_reads_multiple_files(tmp_path):
    _write_triplets_file(tmp_path / "wiki_0_triplets.jsonl", [_row(0)])
    _write_triplets_file(tmp_path / "wiki_1_triplets.jsonl", [_row(1)])

    train_dataset, eval_dataset = train.load_triplet_dataset(str(tmp_path), holdout_files=0)

    assert len(list(train_dataset)) == 2
    assert eval_dataset is None


def test_load_triplet_dataset_holds_out_eval_files(tmp_path):
    _write_triplets_file(tmp_path / "wiki_0_triplets.jsonl", [_row(0), _row(1)])
    _write_triplets_file(tmp_path / "wiki_1_triplets.jsonl", [_row(2)])

    train_dataset, eval_dataset = train.load_triplet_dataset(str(tmp_path), holdout_files=1)

    assert len(list(train_dataset)) == 1  # wiki_0 held out for eval (first file, sorted)
    assert eval_dataset is not None
    assert eval_dataset["anchor"] == ["a0", "a1"]
    assert set(eval_dataset.column_names) == {"anchor", "positive", "negative"}


def test_load_triplet_dataset_raises_when_no_files(tmp_path):
    with pytest.raises(ValueError, match="mine-triplets"):
        train.load_triplet_dataset(str(tmp_path))


def test_train_transfer_model_wires_trainer_correctly(tmp_path, monkeypatch):
    triplets_dir = tmp_path / "triplets"
    triplets_dir.mkdir()
    _write_triplets_file(triplets_dir / "wiki_0_triplets.jsonl", [_row(0)])
    output_dir = tmp_path / "model_out"

    fake_model = MagicMock()
    model_ctor = MagicMock(return_value=fake_model)
    monkeypatch.setattr(train, "SentenceTransformer", model_ctor)

    fake_loss = MagicMock()
    loss_ctor = MagicMock(return_value=fake_loss)
    monkeypatch.setattr(train.losses, "MultipleNegativesRankingLoss", loss_ctor)

    fake_trainer = MagicMock()
    fake_trainer.state.best_model_checkpoint = None
    fake_trainer.state.best_metric = None
    trainer_ctor = MagicMock(return_value=fake_trainer)
    monkeypatch.setattr(train, "SentenceTransformerTrainer", trainer_ctor)

    train.train_transfer_model(
        triplets_dir=str(triplets_dir),
        output_dir=str(output_dir),
        base_model_name="some-base-model",
        batch_size=8,
        gradient_accumulation_steps=2,
        max_steps=100,
        learning_rate=1e-5,
        scale=10.0,
        device="cpu",
    )

    model_ctor.assert_called_once()
    assert model_ctor.call_args[0][0] == "some-base-model"

    loss_ctor.assert_called_once_with(model=fake_model, scale=10.0)

    trainer_ctor.assert_called_once()
    _, trainer_kwargs = trainer_ctor.call_args
    assert trainer_kwargs["model"] is fake_model
    assert trainer_kwargs["loss"] is fake_loss
    assert trainer_kwargs["args"].max_steps == 100
    assert trainer_kwargs["args"].per_device_train_batch_size == 8
    assert trainer_kwargs["args"].gradient_accumulation_steps == 2
    assert trainer_kwargs["args"].fp16 is False  # only enabled on cuda
    # single triplet file -> nothing held out -> no evaluator, eval disabled
    assert trainer_kwargs["evaluator"] is None
    assert trainer_kwargs["args"].eval_strategy.value == "no"
    assert trainer_kwargs["args"].load_best_model_at_end is False

    fake_trainer.train.assert_called_once()
    fake_model.save.assert_called_once_with(str(output_dir))
    assert output_dir.is_dir()  # created up front so TrainingArguments/save have somewhere to write


def test_train_transfer_model_enables_fp16_on_cuda(tmp_path, monkeypatch):
    triplets_dir = tmp_path / "triplets"
    triplets_dir.mkdir()
    _write_triplets_file(triplets_dir / "wiki_0_triplets.jsonl", [_row(0)])

    monkeypatch.setattr(train, "SentenceTransformer", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(train.losses, "MultipleNegativesRankingLoss", MagicMock(return_value=MagicMock()))
    fake_trainer = MagicMock()
    fake_trainer.state.best_model_checkpoint = None
    fake_trainer.state.best_metric = None
    trainer_ctor = MagicMock(return_value=fake_trainer)
    monkeypatch.setattr(train, "SentenceTransformerTrainer", trainer_ctor)

    train.train_transfer_model(
        triplets_dir=str(triplets_dir),
        output_dir=str(tmp_path / "out"),
        device="cuda",
    )

    _, trainer_kwargs = trainer_ctor.call_args
    assert trainer_kwargs["args"].fp16 is True


def test_train_transfer_model_wires_evaluator_when_eval_split_exists(tmp_path, monkeypatch):
    triplets_dir = tmp_path / "triplets"
    triplets_dir.mkdir()
    _write_triplets_file(triplets_dir / "wiki_0_triplets.jsonl", [_row(0)])
    _write_triplets_file(triplets_dir / "wiki_1_triplets.jsonl", [_row(1)])

    monkeypatch.setattr(train, "SentenceTransformer", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(train.losses, "MultipleNegativesRankingLoss", MagicMock(return_value=MagicMock()))

    fake_evaluator = MagicMock()
    evaluator_ctor = MagicMock(return_value=fake_evaluator)
    monkeypatch.setattr(train, "TripletEvaluator", evaluator_ctor)

    fake_trainer = MagicMock()
    fake_trainer.state.best_model_checkpoint = "checkpoint-1"
    fake_trainer.state.best_metric = 0.9
    trainer_ctor = MagicMock(return_value=fake_trainer)
    monkeypatch.setattr(train, "SentenceTransformerTrainer", trainer_ctor)

    train.train_transfer_model(
        triplets_dir=str(triplets_dir),
        output_dir=str(tmp_path / "out"),
        max_steps=10,
        device="cpu",
    )

    evaluator_ctor.assert_called_once()
    _, trainer_kwargs = trainer_ctor.call_args
    assert trainer_kwargs["evaluator"] is fake_evaluator
    assert trainer_kwargs["eval_dataset"] is not None
    assert trainer_kwargs["args"].eval_strategy.value == "steps"
    assert trainer_kwargs["args"].load_best_model_at_end is True
    assert trainer_kwargs["args"].metric_for_best_model == "eval_cosine_accuracy"
