import json
from unittest.mock import MagicMock

import pytest

from swsearch import train


def _write_triplets_file(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_triplet_dataset_streams_anchor_positive_negative_only(tmp_path):
    _write_triplets_file(
        tmp_path / "wiki_0_triplets.jsonl",
        [
            {"anchor": "a1", "positive": "p1", "negative": "n1", "source": "Article A", "url": "http://x"},
            {"anchor": "a2", "positive": "p2", "negative": "n2", "source": "Article B", "url": "http://y"},
        ],
    )

    dataset = train.load_triplet_dataset(str(tmp_path))
    rows = list(dataset)

    assert len(rows) == 2
    assert set(rows[0].keys()) == {"anchor", "positive", "negative"}  # source/url dropped
    assert rows[0] == {"anchor": "a1", "positive": "p1", "negative": "n1"}


def test_load_triplet_dataset_reads_multiple_files(tmp_path):
    _write_triplets_file(tmp_path / "wiki_0_triplets.jsonl", [{"anchor": "a", "positive": "p", "negative": "n", "source": "s", "url": ""}])
    _write_triplets_file(tmp_path / "wiki_1_triplets.jsonl", [{"anchor": "a2", "positive": "p2", "negative": "n2", "source": "s", "url": ""}])

    dataset = train.load_triplet_dataset(str(tmp_path))

    assert len(list(dataset)) == 2


def test_load_triplet_dataset_raises_when_no_files(tmp_path):
    with pytest.raises(ValueError, match="mine-triplets"):
        train.load_triplet_dataset(str(tmp_path))


def test_train_transfer_model_wires_trainer_correctly(tmp_path, monkeypatch):
    triplets_dir = tmp_path / "triplets"
    triplets_dir.mkdir()
    _write_triplets_file(triplets_dir / "wiki_0_triplets.jsonl", [{"anchor": "a", "positive": "p", "negative": "n", "source": "s", "url": ""}])
    output_dir = tmp_path / "model_out"

    fake_model = MagicMock()
    model_ctor = MagicMock(return_value=fake_model)
    monkeypatch.setattr(train, "SentenceTransformer", model_ctor)

    fake_loss = MagicMock()
    loss_ctor = MagicMock(return_value=fake_loss)
    monkeypatch.setattr(train.losses, "TripletLoss", loss_ctor)

    fake_trainer = MagicMock()
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
        margin=2.5,
        device="cpu",
    )

    model_ctor.assert_called_once()
    assert model_ctor.call_args[0][0] == "some-base-model"

    loss_ctor.assert_called_once_with(model=fake_model, triplet_margin=2.5)

    trainer_ctor.assert_called_once()
    _, trainer_kwargs = trainer_ctor.call_args
    assert trainer_kwargs["model"] is fake_model
    assert trainer_kwargs["loss"] is fake_loss
    assert trainer_kwargs["args"].max_steps == 100
    assert trainer_kwargs["args"].per_device_train_batch_size == 8
    assert trainer_kwargs["args"].gradient_accumulation_steps == 2
    assert trainer_kwargs["args"].fp16 is False  # only enabled on cuda

    fake_trainer.train.assert_called_once()
    fake_model.save.assert_called_once_with(str(output_dir))
    assert output_dir.is_dir()  # created up front so TrainingArguments/save have somewhere to write


def test_train_transfer_model_enables_fp16_on_cuda(tmp_path, monkeypatch):
    triplets_dir = tmp_path / "triplets"
    triplets_dir.mkdir()
    _write_triplets_file(triplets_dir / "wiki_0_triplets.jsonl", [{"anchor": "a", "positive": "p", "negative": "n", "source": "s", "url": ""}])

    monkeypatch.setattr(train, "SentenceTransformer", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(train.losses, "TripletLoss", MagicMock(return_value=MagicMock()))
    trainer_ctor = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(train, "SentenceTransformerTrainer", trainer_ctor)

    train.train_transfer_model(
        triplets_dir=str(triplets_dir),
        output_dir=str(tmp_path / "out"),
        device="cuda",
    )

    _, trainer_kwargs = trainer_ctor.call_args
    assert trainer_kwargs["args"].fp16 is True
