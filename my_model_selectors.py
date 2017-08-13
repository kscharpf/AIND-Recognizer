import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
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
        best_model = None
        best_score = float("inf")
        for n_states in range(self.min_n_components, self.max_n_components+1):
            try:
               m = self.base_model(n_states)
               logL = m.score(self.X,self.lengths)

               # how many parameters
               # Each state has a transition probability to every other state (the A matrix) : n_states**2
               # Each state has a distribution for each output symbol (B): n_states * number of symbols
               # since X is many samples then the number of symbols would be the size of the row
               # Each state has an initial distribution (pi) for each emission
               numparam = n_states*n_states + n_states*len(self.X[0]) + n_states*len(self.X[0])
               bic = -2*logL + numparam * np.log(len(self.X))
               if bic < best_score:
                   best_model = m
                   best_score = bic
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        ModelSelector.__init__(self, all_word_sequences, 
                                         all_word_Xlengths, 
                                         this_word,
                                         n_constant,
                                         min_n_components,
                                         max_n_components,
                                         random_state,
                                         verbose)
        self.dic_results = {}

    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        best_score = float("-inf")
        for n_states in range(self.min_n_components, self.max_n_components+1):
            try:
               m = self.base_model(n_states)
               logL = m.score(self.X,self.lengths)

               s = 0.
               nw = 0
               for w in self.words:
                   try:
                       if w != self.this_word:
                           if w not in self.dic_results: 
                               tx,tlen = self.hwords[w]
                               hmm_model = GaussianHMM(n_components=n_states, 
                                                       covariance_type="diag", 
                                                           n_iter=1000,
                                                       random_state=self.random_state, 
                                                       verbose=False).fit(tx, tlen)
                               self.dic_results[w] = hmm_model.score(tx,tlen)
                           s += self.dic_results[w]
                           nw += 1
                   except:
                       pass

               dic = logL - s/nw
               if dic > best_score:
                   best_model = m
                   best_score = dic
            except:
                pass

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        kf = KFold(n_splits=min(len(self.lengths),2))
        best_score = float("-inf")
        best_model = None
        best_num_components = 0
        for n_states in range(self.min_n_components, self.max_n_components+1):
            s_score = 0
            split_cnt = 0
            for train, test in kf.split(self.sequences):
                split_cnt += 1
                try: 
                    X_train,X_train_lengths = combine_sequences(train, self.sequences)
                    X_test,X_test_lengths = combine_sequences(test, self.sequences)
                    hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False)
                    hmm_model.fit(X_train, X_train_lengths)
                    s_score += hmm_model.score(X_test, X_test_lengths)
                except:
                    pass
            if (s_score / split_cnt) > best_score:
                best_score = s_score / split_cnt
                best_num_components = n_states

        best_model = GaussianHMM(n_components=best_num_components, covariance_type="diag", n_iter=1000,
                        random_state=self.random_state, verbose=False)
        best_model.fit(self.X, self.lengths)

        return best_model
