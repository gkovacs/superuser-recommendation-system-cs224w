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
import sqlite3 as sql

import pickle
from collections import Counter

import itertools
#from sklearn.preprocessing import normalize
#import bottleneck

DB_NAME="superuser.sqlite3"
con = sql.connect(DB_NAME)
numUsers=10000

def loadDataframe(queryString):
  dataframe = pd_sql.read_frame(queryString, con)
  #dataframe['CreationDate'] = dataframe['CreationDate'].apply(lambda t: sanetime.time(t).seconds)
  return dataframe

questions = loadDataframe("Select Id as QuestionId, AcceptedAnswerId as AnswerId, OwnerUserId as OwnerId, CreationDate, Score, FavoriteCount, Title, Tags from Posts where PostTypeId=1 and Id in (Select ParentId from Posts where PostTypeId=2)")
answers = loadDataframe("Select Id, ParentId as QuestionId, OwnerUserId as OwnerId, CreationDate, Score from Posts where PostTypeId=2 and OwnerUserId in (Select Id From Users Order by Reputation desc limit "+str(numUsers)+");")
users = loadDataframe("Select Id, Reputation, CreationDate From Users order by Reputation desc limit "+str(numUsers))

usersToQuestionsFileName='usersToQuestions.npz'

questionIndexToId=dict(questionIndexId for questionIndexId in questions['QuestionId'].iteritems())

def loadCSRMatrix(fileName):
  npz = np.load(fileName)
  return sparse.csr_matrix((npz['arr_0'], npz['arr_1'], npz['arr_2']), dtype='float32')

print 'loading CSRMatrix'

usersToQuestions = loadCSRMatrix(usersToQuestionsFileName)

def isoDateToUnixSeconds(isoDate):
  return sanetime.time(isoDate).seconds

def getQuestionDictAndTimeDeltas():
  print 'building question_dict'
  #add (questionID, time) pair to dictionary for O(1) lookup. 
  questionIdDate = questions[['QuestionId','CreationDate']]
  questionIdDate['QuestionCreationDate'] = questionIdDate['CreationDate'].apply(isoDateToUnixSeconds)
  questionIdToTimeDict = dict(zip(questionIdDate.QuestionId, questionIdDate.QuestionCreationDate))

  print 'building time_delta'
  #populate the deltas (question answered time - question asked time in seconds.)
  del questionIdDate['CreationDate']
  answersTimeDeltas = answers[['QuestionId','CreationDate']].merge(questionIdDate, on="QuestionId")
  answersTimeDeltas['CreationDate'] = answersTimeDeltas['CreationDate'].apply(isoDateToUnixSeconds)
  timeDeltas = (answersTimeDeltas['CreationDate']-answersTimeDeltas['QuestionCreationDate']).tolist()

  return (questionIdToTimeDict, timeDeltas)

question_dict, time_delta = getQuestionDictAndTimeDeltas()


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
		prob_vec = [1 for i in range(num_buckets)] #counts.
		time = [i * num_months for i in range(num_buckets)]

		for delta in time_delta:
			bucket_index = delta / bucket_s
			prob_vec[bucket_index] += 1 #add fraction of occurences.
		return (prob_vec, time, bucket_s)

print 'running bucketList'

buckList = bucketList(time_delta, 1000, True)
f = open('buckList', 'w')
pickle.dump(buckList, f)
f.close()
(prob_interval, time, bucket_s) = buckList

print 'populating ranks list'

#@profile
def getRanks():
	ranks = []
	numAnswers = " out of "+str(len(answers.index))
	for i in answers.index:
		if i%10000 == 0:
			print >> sys.stderr, str(i) + numAnswers
		answer_time = sanetime.time(answers.ix[i]['CreationDate']).seconds
		answerer_ID = answers.ix[i]['OwnerId']
		true_question_ID = answers.ix[i]['QuestionId']
		
		#get probabilities of questions with (answer_time_sec and answerer_ID)
		prob_questions_smoothed = usersToQuestions[users['Id'] == answerer_ID].toarray()[0] + 1e-7

		question_scores = []

		#print 'building question scores'
		for j in range(len(prob_questions_smoothed)): #loop through each possible question
			questionId = questionIndexToId[j] #get questionID
			question_t = question_dict[questionId] #get question time.

			delta = answer_time - question_t
			bucket = delta / bucket_s
	
			prob_time = prob_interval[bucket]
	
			#print 'questionId: ' + str(questionId)
			#print 'prob_questions_smoothed[j]: ' + str(prob_questions_smoothed[j])
			#print 'prob_time: ' + str(prob_time)
			question_scores.append((prob_questions_smoothed[j]*prob_time, questionId))

		question_scores = sorted(question_scores,reverse=True)
		#print >> sys.stderr, "answer: " + str(answerer_ID) + " is being processed"

	  	for rank,score_and_question in enumerate(question_scores):
  			#print score_and_question
  			(score, question) = score_and_question
  			print 'rank: ' +str(rank) + ' score: ' + str(score) + ' question: ' + str(question)
			if true_question_ID == question:
    				print rank
      				ranks.append(rank)
      				break
	return ranks

ranks = getRanks()

plt.xlabel('Ranks')
plt.ylabel('Frequency')
plt.title('Histogram of Ranks')
plt.plot(Counter(ranks).keys(),Counter(ranks).values())
plt.savefig('scores_time.png')


print ranks

