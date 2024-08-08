'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
'''
Note:
For grading purpose, all bigrams are represented as word1*-*-*-*word2

Although you may use tuple representations of bigrams within your computation, 
the key of the dictionary itself must be word1*-*-*-*word2 at the end of the computation.
'''

import numpy as np
from collections import Counter
import copy


stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    '''

    frequency = {'pos': Counter(), 'neg': Counter()}


    count = 0
    for texts in train['pos']:
        for t in range(len(texts)):
            count += 1
            if t+1 < len(texts):
                word_pair = (texts[t], texts[t+1])
                frequency['pos'][word_pair[0]+'*-*-*-*'+word_pair[1]] += 1
    
    for texts in train['neg']:
        for t in range(len(texts)):
            count += 1
            if t+1 < len(texts):
                word_pair = (texts[t], texts[t+1])
                frequency['neg'][word_pair[0]+'*-*-*-*'+word_pair[1]] += 1

    return frequency

    



def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    stopwords (set of str):
        - Set of stopwords to be excluded

    Output:
    nonstop (dict of Counters): 
        - nonstop[y][x] = frequency of bigram x in texts of class y,
          but only if neither token in x is a stopword. x is in the format 'word1*-*-*-*word2'
    '''
    nonstop = {'pos': Counter(), 'neg': Counter()}
    count = 0
    frequency_copy = copy.deepcopy(frequency)
    for word_pair  in list(frequency_copy['pos']):
        current = word_pair
        word1 = current.split('*',1)[0]
        word2 = current.rsplit('*',1)[1]
        if (word1 in stopwords and word2 in stopwords):
            del frequency_copy['pos'][word_pair]
    
    for word_pair  in list(frequency_copy['neg']):
        current = word_pair
        word1 = current.split('*',1)[0]
        word2 = current.rsplit('*',1)[1]
        if (word1 in stopwords and word2 in stopwords):
            del frequency_copy['neg'][word_pair]
    nonstop['pos'] = frequency_copy['pos']
    nonstop['neg'] = frequency_copy['neg']
    return nonstop





def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of bigram x in y, where x is in the format 'word1*-*-*-*word2'
          and neither word1 nor word2 is a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary bigram given y


    Important: 
    Be careful that your vocabulary only counts bigrams that occurred at least once
    in the training data for class y.
    '''
    likelihood = {}
    counter_neg_sum = 0
    counter_pos_sum = 0
    count = 1
    for key, value in nonstop.items():
        if (key == 'pos'):
            counter_pos_sum = sum(value.values())

        else:
            counter_neg_sum = sum(value.values())
        

    count  = 0 
    inside_likelihood = {}
    num_biagram_type_pos = len(nonstop['pos'])
    num_biagram_type_neg = len(nonstop['neg'])
    OOV_pos = smoothness / ((counter_pos_sum) + ( smoothness* (num_biagram_type_pos + 1)))
    OOV_neg = smoothness / ((counter_neg_sum) + ( smoothness* (num_biagram_type_neg + 1)))

    #for positive
    for key, value in nonstop['pos'].items():
        inside_likelihood[key] = (value + smoothness) / ((counter_pos_sum) + ( smoothness* (num_biagram_type_pos + 1)))

    inside_likelihood['OOV'] = OOV_pos
    likelihood['pos'] = inside_likelihood
    inside_likelihood = {}
      
    #for negative
    for key, value in nonstop['neg'].items():
        inside_likelihood[key] = (value + smoothness) / ((counter_neg_sum) + ( smoothness* (num_biagram_type_neg + 1)))
    inside_likelihood['OOV'] = OOV_neg
    likelihood['neg'] = inside_likelihood


    return likelihood

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    hypotheses = []
    for index, text in enumerate(texts):
        naive_bayes_pos = np.log(prior)
        naive_bayes_neg = np.log(prior)
        for i in range (len(text)):
            if (i < len(text) - 1):
                word_pair = text[i]+"*-*-*-*"+text[i+1]
                if (text[i] not in stopwords or text[i+1] not in stopwords):
                    if (word_pair not in likelihood['pos']):
                        naive_bayes_pos += np.log(likelihood['pos']['OOV'])
                    else: 
                        naive_bayes_pos += np.log(likelihood['pos'][word_pair])
                    if (word_pair not in likelihood['neg']):
                        naive_bayes_neg += np.log(likelihood['neg']['OOV'])
                    else:
                        naive_bayes_neg +=  np.log(likelihood['neg'][word_pair])
        if (naive_bayes_pos > naive_bayes_neg):
            hypotheses.append('pos')
        elif (naive_bayes_pos  < naive_bayes_neg):
            hypotheses.append('neg')
        else:
            hypotheses.append('undecided')
    return hypotheses



def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    # raise RuntimeError("You need to write this part!")
    
    accuracies = np.zeros((len(priors), len(smoothnesses)))
    for idx, m in enumerate(priors):
        hypotheses = []
        for jdx, n in enumerate(smoothnesses):
            likelihood = laplace_smoothing(nonstop, n)
            hypotheses = naive_bayes(texts, likelihood, m)
            accuracy = 0
            for i in range(len(labels)):
                if (labels[i] == hypotheses[i]):
                    accuracy += 1
            accuracy = accuracy / len(labels)
            accuracies[idx, jdx] = accuracy

    return accuracies
                          