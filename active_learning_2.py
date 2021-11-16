import logging
import pandas as pd
import numpy as np

from transformers import AutoTokenizer
from small_text.active_learner import PoolBasedActiveLearner
from small_text.initialization import random_initialization_balanced
from small_text.integrations.transformers import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from small_text.query_strategies import RandomSampling,LeastConfidence,EmptyPoolException,PredictionEntropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from small_text.integrations.transformers.datasets import TransformersDataset

TRANSFORMER_MODEL = TransformerModelArguments('allenai/scibert_scivocab_uncased')


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    print('Train accuracy: {:.2f}'.format(
        f1_score(y_pred, train.y, average='micro')))
    print('Test accuracy: {:.2f}'.format(f1_score(y_pred_test, test.y, average='micro')))
    print('---')

def preprocess_data(tokenizer, data, labels, max_length=500):

    data_out = []

    for i, doc in enumerate(data):
        encoded_dict = tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            return_attention_mask=True,
            return_tensors='pt',
            truncation='longest_first'
        )

        data_out.append((encoded_dict['input_ids'], encoded_dict['attention_mask'], labels[i]))

    return TransformersDataset(data_out)


def active_learning_main():
    # Active learning parameters
    clf_scibert_sci = TransformerBasedClassificationFactory(TRANSFORMER_MODEL)
    query_strategy = LeastConfidence()
    #     query_strategy= PredictionEntropy()

    # Prepare so
    train = pd.read_csv("data/initial_labeled_dataset/other_data.csv",
                        encoding='latin-1').sample(frac=1).drop_duplicates()
    test = pd.read_csv("data/initial_labeled_dataset/testing_data.csv", encoding='latin-1').sample(
        frac=1).drop_duplicates()

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL.model, cache_dir='./')
    x_train = preprocess_data(tokenizer, train["Text"], train["Label"].to_numpy())
    y_train = train["Label"].to_numpy()

    x_test = preprocess_data(tokenizer, test["Text"], test["Label"].to_numpy())
    y_test = test["Label"].to_numpy()

    # Active learner
    active_learner = PoolBasedActiveLearner(clf_scibert_sci, query_strategy, x_train)
    labeled_indices = initialize_active_learner(active_learner, y_train)

    evaluate(active_learner, x_train[labeled_indices], x_test)

    try:
        perform_active_learning(active_learner, x_train, labeled_indices, x_test)

    except PoolExhaustedException as e:
        print('Error! Not enough samples left to handle the query.')
    except EmptyPoolException as e:
        print('Error! No more samples left. (Unlabeled pool is empty)')


def perform_active_learning(active_learner, train, labeled_indices, test):
    # Perform 10 iterations of active learning...
    for i in range(10):
        # ...where each iteration consists of labelling 10 samples
        q_indices = active_learner.query(num_samples=10)

        y =active_learner.classifier.predict(train[q_indices])

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        labeled_indices = np.concatenate([q_indices, labeled_indices])

        print('Iteration #{:d} ({} samples)'.format(i, len(labeled_indices)))
        evaluate(active_learner, train[labeled_indices], test)


def initialize_active_learner(active_learner, y_train):
    x_indices_initial = random_initialization_balanced(y_train, n_samples=143)

    y_initial = np.array([y_train[i] for i in x_indices_initial])

    active_learner.initialize_data(x_indices_initial, y_initial, 4)
    return x_indices_initial


logging.getLogger('small_text').setLevel(logging.INFO)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)


if __name__ == '__main__':
    active_learning_main()

