import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for word_idx in range(test_set.num_items):
        word_probability = {}
        max_log_likelihood = -np.inf
        word_with_max_logl = None
        for word, model in models.items():
            Xlengths = test_set.get_item_Xlengths(word_idx)
            log_likelihood = -np.inf
            try:
                log_likelihood = model.score(Xlengths[0], Xlengths[1])
            except (Exception, ValueError) as e:
                # skip over these words, don't consider scores for them
                pass
            word_probability[word] = log_likelihood
            if log_likelihood > max_log_likelihood:
                max_log_likelihood = log_likelihood
                word_with_max_logl = word
        probabilities.append(word_probability)
        guesses.append(word_with_max_logl)
    return probabilities, guesses
