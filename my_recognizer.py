import warnings
import sys
from asl_data import SinglesData


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
    pidx = 0
    for word_id in range(0, len(test_set.get_all_sequences())):
        X, lengths = test_set.get_item_Xlengths(word_id)
        maxScore = float("-inf")
        bestWord = ""
        probabilities.append(dict())
        for w,m in models.items():
            try:
                logL = m.score(X, lengths)
                if logL > maxScore:
                    maxScore = logL
                    bestWord = w
                probabilities[pidx][w] = logL
            except:
                pass
        guesses.append(bestWord)
        pidx+=1
    return probabilities,guesses
