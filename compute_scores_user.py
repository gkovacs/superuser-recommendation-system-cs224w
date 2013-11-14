#compute_score_user.py

#computes scores.

from features import *

import sys
import math
import sanetime

import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import pandas.io.sql as pd_sql
from scipy import sparse

import itertools
#from sklearn.preprocessing import normalize
#import bottleneck

usersToQuestionsFileName='usersToQuestions.npz'

def loadCSRMatrix(fileName):
  npz = np.load(fileName)
  return sparse.csr_matrix((npz['arr_0'], npz['arr_1'], npz['arr_2']), dtype='float32')

print 'loading CSRMatrix'

usersToQuestions = loadCSRMatrix(usersToQuestionsFileName)

print 'populating ranks list'

ranks = []

for i in answers.index:
	answerer_ID = answers.ix[i][2]
	true_question_ID = answers.ix[i][1]
	#get probabilities of questions with (answer_time_sec and answerer_ID)
	prob_questions = usersToQuestions[users['Id'] == answerer_ID].todense().tolist()
	prob_questions_smoothed = [prob +1e-7 for prob in prob_questions[0]]

	question_scores = []

	for j in range(len(prob_questions_smoothed)): #loop through each possible question
		question_scores.append((prob_questions_smoothed[j], questionId))

	question_scores = sorted(question_scores,reverse=True)
	print >> sys.stderr, "answer: " + str(answerer_ID) + "is being processed"
  	for rank,score_and_question in enumerate(question_scores):
  		(score, question) = score_and_question
    	if true_question_ID == question:
      		ranks.append(rank)
      		break

print ranks

	

	

	
