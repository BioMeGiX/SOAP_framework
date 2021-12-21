import pandas as pd
import argparse

# Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
import tensorflow_hub as hub

from tensorflow.keras import models
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import SeparableConv1D

import ktrain
from ktrain import text as ktrain_text
# SKLearn & bert
import bert
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import numpy as np
from text_preprocesssing import  normalize_corpus

max_seq_length = 128  # Your choice here.
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])


topk = 14000
FLAGS = None

MAX_SEQUENCE_LENGTH = 500


epochs=10
learning_rate=8e-5
batch_size=6
filters=64
blocks=2
dropout_rate=0.3
embedding_dim=400
kernel_size=3
pool_size=3


def load_csv_dataset():
    data_training_file = "data/final_dataset/training_data.csv"
    data_testing_file = "data/final_dataset/testing_data.csv"

    training_data = pd.read_csv(data_training_file, encoding='latin-1').sample(frac=1).drop_duplicates()
    testing_data = pd.read_csv(data_testing_file, encoding='latin-1').sample(frac=1).drop_duplicates()

    target_names = ["Assessment", "Subjective", "Objective", "Plan"]

    train_texts = training_data["Text"]
    train_labels = training_data["Label"]

    val_texts = testing_data["Text"]
    val_labels = testing_data["Label"]

    return ((train_texts, train_labels), (val_texts, val_labels), target_names)


def vectorize_by_sequence(train_texts, val_texts):
    tokenizer = text.Tokenizer(num_words=topk)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index


def convert_to_tensor(train_texts, val_texts):
    x_train = tf.convert_to_tensor(train_texts, dtype=tf.string)
    x_val = tf.convert_to_tensor(val_texts, dtype=tf.string)
    return x_train, x_val


def bert_tokenizer(texts):
    FullTokenizer = bert.bert_tokenization.FullTokenizer
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    tokens = tokenizer.tokenize(texts)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return token_ids


def convert_to_tensor(train_texts, val_texts):
    x_train = tf.convert_to_tensor(train_texts, dtype=tf.string)
    x_val = tf.convert_to_tensor(val_texts, dtype=tf.string)
    return x_train, x_val


def bert_tokenizer(texts):
    FullTokenizer = bert.bert_tokenization.FullTokenizer
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    tokens = tokenizer.tokenize(texts)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return token_ids



((train_texts, train_labels), (val_texts,val_labels),target_names) = load_csv_dataset()

normaliz_train_text= normalize_corpus(train_texts)
normaliz_val_texts=normalize_corpus(val_texts)

x_train, x_val, word_index = vectorize_by_sequence(normaliz_train_text, normaliz_val_texts)

bert_tokenized_train_text = [bert_tokenizer(train_text) for train_text in normaliz_train_text]

num_classes =len(target_names)

input_shape=x_train.shape[1:]

# reserved index 0.
num_features = min(len(word_index) + 1, topk)


if num_classes == 2:
    loss = 'binary_crossentropy'
else:
    loss = 'sparse_categorical_crossentropy'
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)






def get_lastlayer_activation_function(num_classes):

    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def train_svm_model():
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    svm_model = text_clf.fit(train_texts, train_labels)

    return svm_model


svm_model= train_svm_model()

predicted = svm_model.predict(val_texts)

print(metrics.classification_report(val_labels, predicted,
    target_names=target_names))


def train_cnn_model():
    op_units, op_activation = get_lastlayer_activation_function(num_classes)

    model = models.Sequential()

    model.add(Embedding(input_dim=num_features, output_dim=embedding_dim, input_length=input_shape[0]))

    for _ in range(blocks - 1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(op_units, activation=op_activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val, val_labels),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    #     model.save('soap_sepcnn_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]


def train_rnn_model():
    op_units, op_activation = get_lastlayer_activation_function(num_classes)

    model = models.Sequential()
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_shape[0]))

    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(5, 10)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(op_units, activation=op_activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    model.summary()

    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        validation_data=(x_val, val_labels),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    #     model.save('soap_sepcnn_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]


def train_biLstm_model():
    op_units, op_activation = get_lastlayer_activation_function(num_classes)
    model = models.Sequential()
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_shape[0]))

    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))
    model.add(Dense(op_units, activation=op_activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    model.summary()

    history = model.fit(
        bert_tokenized_train_text,
        train_labels,
        epochs=epochs,
        validation_data=(x_val, val_labels),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    #     model.save('soap_sepcnn_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]


def train_bert_model(model_name):
    MODEL_NAME = model_name

    t = ktrain_text.Transformer(MODEL_NAME, maxlen=MAX_SEQUENCE_LENGTH, class_names=target_names)
    trn = t.preprocess_train(normaliz_train_text.to_numpy(), train_labels.to_numpy())
    val = t.preprocess_test(normaliz_val_texts.to_numpy(), val_labels.to_numpy())
    model = t.get_classifier()
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=batch_size)

    learner.fit_onecycle(learning_rate, epochs)

    learner.validate(class_names=target_names)


if __name__ == '__main__':
    # train_cnn_model()
    # train_rnn_model()
    # train_biLstm_model()
    train_bert_model("allenai/scibert_scivocab_uncased")





