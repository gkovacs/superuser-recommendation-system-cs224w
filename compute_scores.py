#computes scores.

#from features import *

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

question_dict = dict()

print 'building question_dict'

for i in questions.index:
	time_in_sec = sanetime.time(questions.ix[i][3]).seconds
	question_dict[questions.ix[i][0]] = time_in_sec
	#add (answerID, time) pair to dictionary for O(1) lookup. 

print 'buildling time_delta'

#populate the deltas (question answered time - question asked time in seconds.)
time_delta = []

for i in answers.index:
	question_t = question_dict[answers.ix[i][1]] #time question was asked.
	answered_t = sanetime.time(answers.ix[i][3]).seconds #time answered.
	delta = answered_t - question_t
	time_delta.append(delta)

def bucketList(time_delta, num_buckets, normalize):
	time_min = 0
	time_max = max(time_delta)

	spread = time_max + 1
	#lower bound is time_min-1 and upper is time_max+1

	bucket_s = spread / num_buckets

	if spread % num_buckets != 0: #last bucket is left out because of int division.
		bucket_s += 1

	num_months = bucket_s / (3600 * 24.0 * 30)

	norm_const = len(time_delta) + num_buckets

	if normalize:
		prob_vec = [1.0 / norm_const for i in range(num_buckets)] #smoothing adding 1/norm_const to each bucket.
		time = [i * num_months for i in range(num_buckets)]

		for delta in time_delta:
			bucket_index = delta / bucket_s
			prob_vec[bucket_index] += 1.0 / norm_const #add fraction of occurences.
		return (prob_vec, time, bucket_s)
	else:
		prob_vec = [1 for i in range(num_buckets)] #smoothing adding 1/norm_const to each bucket.
		time = [i * num_months for i in range(num_buckets)]

		for delta in time_delta:
			bucket_index = delta / bucket_s
			prob_vec[bucket_index] += 1 #add fraction of occurences.
		return (prob_vec, time, bucket_s)

print 'running bucketList'

(prob_interval, time, bucket_s) = bucketList(time_delta, 1000, True)

print 'populating ranks list'

ranks = []

#for i in answers.index:
for i in range(100):
	print >> sys.stderr, str(i) + " out of 300"# + str(len(answers.index)) 
	answer_time = sanetime.time(answers.ix[i][3]).seconds
	answerer_ID = answers.ix[i][2]
	true_question_ID = answers.ix[i][1]
	#get probabilities of questions with (answer_time_sec and answerer_ID)
	prob_questions = usersToQuestions[users['Id'] == answerer_ID].todense().tolist()
	prob_questions_smoothed = [prob +1e-7 for prob in prob_questions[0]]

	question_scores = []

	for j in range(len(prob_questions_smoothed)): #loop through each possible question
		questionId = questions['QuestionId'].iloc[j] #get questionID
		question_t = question_dict[questionId] #get question time.

		delta = answer_time - question_t
		bucket = delta / bucket_s

		prob_time = prob_interval[bucket]

		question_scores.append((prob_questions_smoothed[j]*prob_time, questionId))

	question_scores = sorted(question_scores,reverse=True)
	print >> sys.stderr, "answer: " + str(answerer_ID) + "is being processed"
  	for rank,score_and_question in enumerate(question_scores):
  		(score, question) = score_and_question
    	if true_question_ID == question:
    		print rank
      		ranks.append(rank)
      		break

plt.xlabel('Ranks')
plt.ylabel('Frequency')
plt.title('Histogram of Ranks')
plt.hist(ranks)
plt.savefig('scores.png')


print ranks

	

	

	
