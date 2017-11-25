import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

from typing import Optional
import traceback

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3, n_folds=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.n_folds = n_folds
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self) -> Optional[GaussianHMM]:
        raise NotImplementedError

    def base_model(self, num_states) -> Optional[GaussianHMM]:
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        competing_models = []
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                if model:
                    competing_models.append(model)
            except Exception as e:
                if self.verbose:
                    print(e)
                pass

        min_bic_score = np.inf
        best_model = None
        for model in competing_models:
            try:
                bic_score = self.calculate_bic_score(model.score(self.X, self.lengths),
                                         self.no_of_free_parameters(model.n_components, len(self.X[0])),
                                         len(self.X[:,0]))
                if bic_score < min_bic_score:
                    min_bic_score = bic_score
                    best_model = model
            except Exception as e:
                if self.verbose:
                    # traceback.print_exc()
                    print(e)
                pass
        return best_model

    def no_of_free_parameters(self, no_of_states: int, no_of_features: int) -> int:
        '''
        Calculates the number of free parameters (to be estimated) by model fitting
        :param no_of_states: an integer 
            Number of states of the HMM
        :param no_of_features: an integer
            Number of features used for training the HMM
            
        The formula for calculating number of free parameters is as follows
            number_of_free_parameters(P) = initial_state_probabilities + transition_probabilities + emission_probability_gaussian_descriptors
            => P = (no_of_states - 1) + (no_of_states * (no_of_states - 1)) + (no_of_features * no_of_states * 2) (1 + 1 for mean & SD)
            => P = no_of_states ^ 2 - 1 + no_of_features * no_of_states * 2
            
        References:
            1) https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/11
        :return: no_of_free_parameters
        '''
        return no_of_states ** 2 - 1 + (no_of_features * no_of_states * 2)

    def calculate_bic_score(self, log_likelihood: float, no_of_parameters: int, no_of_data_points: int):
        return -2 * log_likelihood + no_of_parameters * np.log(no_of_data_points)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        competing_models = []
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                if model:
                    competing_models.append(model)
            except Exception as e:
                if self.verbose:
                    print(e)
                pass
        all_words = list(self.words.keys())
        all_other_words = [word for word in all_words if word != self.this_word]
        max_dic_score = -np.inf
        best_model = None
        for model in competing_models:
            try:
                this_word_log_likelihood = model.score(self.X, self.lengths)
                all_other_words_log_likelihood = [model.score(self.hwords[word][0], self.hwords[word][1]) for word in all_other_words]
                avg_all_other_words_log_likelihood = np.mean(all_other_words_log_likelihood)
                dic_score = this_word_log_likelihood - avg_all_other_words_log_likelihood
                if dic_score > max_dic_score:
                    max_dic_score = dic_score
                    best_model = model
            except (Exception, ValueError) as e:
                if self.verbose:
                    # traceback.print_exc()
                    print(e)
                pass
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        kf = KFold(n_splits=self.n_folds)
        model_scores = []
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                cur_model_log_likelihoods = []
                model = None
                if len(self.sequences) > self.n_folds:
                    for train_index, test_index in kf.split(self.sequences):
                        if self.verbose:
                            print("Train fold indices:{} Test fold indices:{} No of components:{}".format(train_index, test_index, n))
                        self.X, self.lengths = combine_sequences(train_index, self.sequences)
                        test_X, test_lengths = combine_sequences(test_index, self.sequences)
                        model = self.base_model(n)
                        if model:
                            log_likelihood = model.score(test_X, test_lengths)
                            cur_model_log_likelihoods.append(log_likelihood)
                    model_scores.append((model, np.mean(cur_model_log_likelihoods)))
                else:
                    model = self.base_model(n)
                    model_scores.append((model, model.score(self.X, self.lengths)))
            except Exception as e:
                if self.verbose:
                    # traceback.print_exc()
                    print(e)
                pass

        if model_scores:
            return sorted(model_scores, key = lambda x: x[1], reverse = True)[0][0]
        else:
            return None
