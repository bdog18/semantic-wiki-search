"""Research spike: a custom transformer encoder trained from scratch with
triplet loss, for the baseline / transfer-learning / custom-transformer
3-model comparison -- see README.md. Not wired into the swsearch CLI; run
directly (`python research/custom_encoder.py --help`) so TensorFlow/Keras
stay an opt-in research dependency (requirements-research.txt) rather than a
hard dependency of the installable package.

Reuses the same triplets file mined once against the baseline index (`swsearch
mine-triplets`) -- no separate mining run is needed for this model either.
"""

import argparse
import json
import os

import tensorflow as tf
from keras import Model, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from swsearch.config import settings


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))


class CustomEncoder(Model):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers=2):
        super().__init__()
        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers + 1)]
        self.pooling = layers.GlobalAveragePooling1D()

    def call(self, x, training=False):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return self.pooling(x)


def triplet_loss(anchor, positive, negative, margin=0.3):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)


class TripletTrainer(Model):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        anchor, positive, negative = data
        with tf.GradientTape() as tape:
            a_embed = self.encoder(anchor, training=True)
            p_embed = self.encoder(positive, training=True)
            n_embed = self.encoder(negative, training=True)
            loss = triplet_loss(a_embed, p_embed, n_embed)

        grads = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def create_vectorizer(input_dir, vectorizer_path, vocab_size, max_len):
    print("Fitting vectorizer on all data (streamed)")
    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
    )

    def text_generator(input_dir):
        files = sorted(f for f in os.listdir(input_dir) if f.endswith(".jsonl"))
        for part in tqdm(files, desc="yielding text"):
            with open(os.path.join(input_dir, part), "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    yield item["anchor"]
                    yield item["positive"]
                    yield item["negative"]

    text_ds = tf.data.Dataset.from_generator(
        lambda: text_generator(input_dir),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string),
    ).batch(2048)

    vectorizer.adapt(text_ds)

    print("Saving vectorizer")
    model_for_saving = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer,
    ])
    model_for_saving.save(vectorizer_path)


def load_triplet_dataset_streamed(json_dir, vectorizer, batch_size):
    def generator():
        for file in sorted(os.listdir(json_dir)):
            if not file.endswith(".jsonl"):
                continue
            with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        t = json.loads(line)
                        a, p, n = t["anchor"].strip(), t["positive"].strip(), t["negative"].strip()
                        if all(len(x.split()) > 3 for x in (a, p, n)):
                            yield a, p, n
                    except Exception:
                        continue

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    def vectorize_fn(anchor, pos, neg):
        return vectorizer(anchor), vectorizer(pos), vectorizer(neg)

    return (
        dataset.map(vectorize_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets-dir", default=str(settings.paths.triplets_dir), help="Mined triplet JSONL directory (from `swsearch mine-triplets`).")
    parser.add_argument("--output-dir", default=str(settings.paths.custom_model_dir), help="Where to save the vectorizer and encoder weights.")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2, help="EarlyStopping patience, in epochs, on training loss.")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    vectorizer_dir = os.path.join(args.output_dir, "saved_vectorizer")
    weights_path = os.path.join(args.output_dir, "best_encoder.weights.h5")
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(vectorizer_dir):
        print("Loading saved vectorizer")
        vectorizer = tf.keras.models.load_model(vectorizer_dir)
    else:
        create_vectorizer(args.triplets_dir, vectorizer_dir, args.vocab_size, args.max_len)
        vectorizer = tf.keras.models.load_model(vectorizer_dir)

    # Not .repeat()'d: each epoch is one full pass over the (unknown-length,
    # streamed) triplets file. tf.data/Keras handle an epoch ending on
    # generator exhaustion natively, so there's no need to know the triplet
    # count up front the way a fixed steps_per_epoch would -- that count is
    # corpus-dependent (varies with however many triplets this run's corpus
    # produced) and was previously a stale hardcoded value from an earlier,
    # much smaller run.
    train_dataset = load_triplet_dataset_streamed(args.triplets_dir, vectorizer, args.batch_size)

    encoder = CustomEncoder(args.vocab_size, args.max_len, args.embed_dim, args.num_heads, args.ff_dim)
    if os.path.exists(weights_path):
        print("Loading best weights")
        encoder.load_weights(weights_path)
    trainer = TripletTrainer(encoder)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))

    callbacks = [
        EarlyStopping(monitor="loss", patience=args.patience),
        ModelCheckpoint(
            filepath=weights_path,
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]
    trainer.fit(
        train_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print(f"Training complete and weights saved to {weights_path}.")
