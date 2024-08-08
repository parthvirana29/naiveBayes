'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''
    # raise RuntimeError("You need to write this part!")
    #used ChatGPT

    frequency = {}
    for t in texts:
      count_word0 = t.count(word0)
      if (count_word0 in frequency):
        frequency[count_word0] += 1
      else:
        frequency[count_word0] = 1
    
    max_count_word0 = max(frequency.keys())

    Pmarginal = np.zeros(max_count_word0 + 1)

    for count, freq in frequency.items():
      Pmarginal[count] = freq/len(texts)

    return Pmarginal
    
def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''
    # raise RuntimeError("You need to write this part!")
    #used ChatGPT

    freq_word0 = {}
    freq_word1 = {}
    freq_joint = {}

    for t in texts:
      count_0 = t.count(word0)
      count_1 = t.count(word1)
      if count_0 in freq_word0:
        freq_word0[count_0] += 1
      else:
        freq_word0[count_0] = 1
      
      if count_1 in freq_word1:
        freq_word1[count_1] +=1 
      else:
        freq_word1[count_1] = 1

      joint_count = (count_0, count_1)
      if joint_count in freq_joint:
        freq_joint[joint_count] += 1
      else:
        freq_joint[joint_count] = 1
      
      max_word0 = max(freq_word0.keys())
      max_word1 = max(freq_word1.keys())

    Pcond = np.zeros((max_word0 + 1, max_word1 + 1))
    for x0 in range(max_word0 + 1):
      for x1 in range(max_word1 + 1):
        curr = (x0,x1)
        if (curr in freq_joint):
          Pcond[x0,x1] = freq_joint[curr] /  freq_word0[x0]
    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''
    # raise RuntimeError("You need to write this part!")
    Pjoint = np.zeros((len(Pcond), len(Pcond[0])))
    for x0 in range(len(Pcond)):
      for x1 in range(len(Pcond[0])):
          Pjoint[x0][x1] = Pcond[x0][x1] * Pmarginal[x0]
    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''
    # raise RuntimeError("You need to write this part!")
    mu = np.zeros(2)
    for x0 in range(len(Pjoint)):
      for x1 in range(len(Pjoint[0])):
        if not np.isnan(Pjoint[x0, x1]):
          mu[0] += x0*Pjoint[x0,x1]
          mu[1] += x1*Pjoint[x0,x1]    
    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    # raise RuntimeError("You need to write this part!")
    Sigma = np.zeros((len(mu), len(mu)))

    # Calculate the covariance matrix elements
    for x0 in range(Pjoint.shape[0]):
      for x1 in range(Pjoint.shape[1]):
        if not np.isnan(Pjoint[x0, x1]):
          Sigma[0,0] += ((x0 - mu[0]) * (x0-mu[0])) *Pjoint[x0,x1]
          Sigma[0,1] += ((x0 - mu[1]) * (x1- mu[1])) * Pjoint[x0,x1]
          Sigma[1,0] += ((x1 - mu[1])) * (x0-mu[0]) * Pjoint[x0,x1]
          Sigma[1,1] += ((x1 - mu[1])) * (x1-mu[1]) *Pjoint[x0,x1]
          #different approach


    return Sigma
    

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    # raise RuntimeError("You need to write this part!")
    temp = {}
    Pfunc = {}

    for x0 in range(len(Pjoint)):
      for x1 in range(len(Pjoint[0])):
        if (f(x0,x1) in temp):
          temp[f(x0,x1)] += Pjoint[x0,x1]
        else:
          temp[f(x0,x1)] = Pjoint[x0,x1]
    
    total = sum(temp.values())
    for key, val in temp.items():
      Pfunc[key] = val / total
      
    return Pfunc
    

