"""
Created by Christos Baziotis.
modified by Weiheng Li
"""
import random

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from kutilities.helpers.data_preparation import print_dataset_statistics, \
    labels_to_categories, categories_to_onehot
from sklearn.cross_validation import train_test_split

from dataset.data_loader import SemEvalDataLoader
from sk_transformers.CustomPreProcessor import CustomPreProcessor
from sk_transformers.EmbeddingsExtractor import EmbeddingsExtractor
from ignore_warnings import set_ignores

set_ignores()
from sklearn.pipeline import Pipeline
import numpy
numpy.set_printoptions(threshold=numpy.inf)
from WordVectorsManager import WordVectorsManager


def prepare_dataset(X, y, pipeline, y_one_hot=True, y_as_is=False):
    try:
        print_dataset_statistics(y)
    except:
        pass

    X = pipeline.fit_transform(X)

    if y_as_is:
        try:
            return X, numpy.asarray(y, dtype=float)
        except:
            return X, y

    # 1 - Labels to categories
    y_cat = labels_to_categories(y)

    if y_one_hot:
        # 2 - Labels to one-hot vectors
        return X, categories_to_onehot(y_cat)

    return X, y_cat


class Task4Loader:
    """
    Task 4: Sentiment Analysis in Twitter
    """

    def __init__(self, word_indices, text_lengths, subtask="A", silver=False,
                 **kwargs):

        self.word_indices = word_indices

        filter_classes = kwargs.get("filter_classes", None)
        self.y_one_hot = kwargs.get("y_one_hot", True)

        self.pipeline = Pipeline([
            ('preprocess', CustomPreProcessor(TextPreProcessor(
                backoff=['url', 'email', 'percent', 'money', 'phone', 'user',
                         'time', 'url', 'date', 'number'],
                include_tags={"hashtag", "allcaps", "elongated", "repeated",
                              'emphasis', 'censored'},
                fix_html=True,
                segmenter="twitter",
                corrector="twitter",
                unpack_hashtags=True,
                unpack_contractions=True,
                spell_correct_elong=False,
                tokenizer=SocialTokenizer(lowercase=True).tokenize,
                dicts=[emoticons]))),
            ('ext', EmbeddingsExtractor(word_indices=word_indices,
                                        max_lengths=text_lengths,
                                        add_tokens=(False,
                                                    True) if subtask != "A" else True,
                                        unk_policy="random"))])

        # loading data
        print("Loading data...")
        dataset = SemEvalDataLoader(verbose=False).get_data(task=subtask,
                                                            years=None,
                                                            datasets=None,
                                                            only_semeval=True)
        random.Random(42).shuffle(dataset)

        if filter_classes:
            dataset = [d for d in dataset if d[0] in filter_classes]

        self.X = [obs[1] for obs in dataset]
        self.y = [obs[0] for obs in dataset]
        print("total observations:", len(self.y))

        print("-------------------\ntraining set stats\n-------------------")
        print_dataset_statistics(self.y)
        print("-------------------")

        if silver:
            print("Loading silver data...")
            dataset = SemEvalDataLoader().get_silver()
            self.silver_X = [obs[1] for obs in dataset]
            self.silver_y = [obs[0] for obs in dataset]
            print("total observations:", len(self.silver_y))

    def load_final(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=0.1,
                                                            stratify=self.y,
                                                            random_state=27)
        print("\nPreparing training set...")
        training = prepare_dataset(X_train, y_train, self.pipeline,
                                   self.y_one_hot)
        print("\nPreparing test set...")
        testing = prepare_dataset(X_test, y_test, self.pipeline, self.y_one_hot)
        return training, testing
