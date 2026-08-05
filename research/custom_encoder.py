"""Archived research spike: a custom transformer encoder trained with triplet
loss, meant to eventually replace the off-the-shelf SentenceTransformer model.

Deprioritized relative to the working baseline (whole-article SentenceTransformer
embeddings + flat FAISS index + cosine rerank) -- see README.md. Training was
never completed (no best_encoder.weights.h5 / saved_vectorizer/ exist), and it
is not wired into the swsearch CLI. Kept here, outside src/swsearch/, so
TensorFlow/Keras stay an opt-in research dependency (requirements-research.txt)
rather than a hard dependency of the installable package.
"""

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


if __name__ == '__main__':
    input_dir = str(settings.paths.triplets_dir)
    vectorizer_dir = str(settings.paths.custom_model_dir / "saved_vectorizer")
    weights_dir = str(settings.paths.custom_model_dir)
    vocab_size = 30000
    max_len = 32
    embed_dim = 128
    num_heads = 8
    ff_dim = 256
    batch_size = 512
    num_epochs = 30
    learning_rate = 1e-4
    total_lines = 15123359

    if os.path.exists(vectorizer_dir):
        print("Loading saved vectorizer")
        vectorizer = tf.keras.models.load_model(vectorizer_dir)
    else:
        create_vectorizer(input_dir, vectorizer_dir, vocab_size, max_len)
        vectorizer = tf.keras.models.load_model(vectorizer_dir)

    train_dataset = load_triplet_dataset_streamed(input_dir, vectorizer, batch_size)

    encoder = CustomEncoder(vocab_size, max_len, embed_dim, num_heads, ff_dim)
    weights_path = f"{weights_dir}/best_encoder.weights.h5"
    if os.path.exists(weights_path):
        print("Loading best weights")
        encoder.load_weights(weights_path)
    trainer = TripletTrainer(encoder)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

    callbacks = [
        EarlyStopping(monitor="loss", patience=2),
        ModelCheckpoint(
            filepath=weights_path,
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]
    trainer.fit(
        train_dataset.repeat(),
        steps_per_epoch=total_lines // batch_size,
        epochs=num_epochs,
        callbacks=callbacks,
    )

    print("Training complete and weights saved.")
