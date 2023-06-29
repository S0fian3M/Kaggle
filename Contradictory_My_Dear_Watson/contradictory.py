import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import matplotlib.pyplot as plt
import tensorflow as tf


def encode_sentence(sentence: str, tokenizer):
    """
    Encode a sentence
    :param sentence:
    :return:
    """
    tokens = list(tokenizer.tokenize(sentence))
    tokens.append('[SEP]') # Sentence separator token
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(
        hypotheses,
        premises,
        tokenizer
):
    """
    Encode dataset
    :param hypotheses: First part of the input
    :param premises: Second part of the input
    :param tokenizer:
    :return:
    """
    num_examples = len(hypotheses)

    sentence1 = tf.ragged.constant([encode_sentence(s, tokenizer) for s in np.array(hypotheses)])
    sentence2 = tf.ragged.constant([encode_sentence(s, tokenizer) for s in np.array(premises)])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat(
        [type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs


def build_model(
        bert_name: str,
        max_len=50
):
    """
    Build the model
    :param bert_name:
    :param max_len:
    :return:
    """
    bert_encoder = TFBertModel.from_pretrained(bert_name)
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")

    embedding = bert_encoder([input_word_ids, input_mask, input_type_ids])[0]
    # Feed forward network
    x = tf.keras.layers.Dense(1024, activation="relu")(embedding[:, 0, :])
    x = tf.keras.Dense(256, activation="relu")(x)
    output = tf.keras.layers.Dense(3, activation='softmax')(x) # 3 Classes

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)

train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

labels, frequencies = np.unique(train_data.language.values, return_counts=True)

plt.figure(figsize=(10,10))
plt.pie(frequencies, labels=labels, autopct='%1.1f%%')
plt.savefig("language_distribution.svg", format="svg")
plt.show()

bert_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(bert_name)

train_input = bert_encode(train_data.premise.values, train_data.hypothesis.values, tokenizer)
with strategy.scope():
    model = build_model(bert_name)
    model.summary()

model.fit(train_input, train_data.label.values, epochs=2, batch_size=64, validation_split=0.3)

test_input = bert_encode(test_data.premise.values, test_data.hypothesis.values, tokenizer)

predictions = [np.argmax(i) for i in model.predict(test_input)]

output = pd.DataFrame({'prediction': predictions})
output.to_csv('submission.csv', index=False)
