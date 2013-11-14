#!/usr/bin/python
# Arpad Kovacs <akovacs@stanford.edu>
# CS224W Final Project - Feature Importer

#import snap as snap
#import networkx as nx
#from collections import Counter
#from pylab import *
#import matplotlib.pyplot as pyplot
#import random

import pandas as pd
import pandas.io.sql as pd_sql
import numpy as np
from scipy import sparse
#from sklearn.preprocessing import normalize
#import bottleneck

import matplotlib.pyplot as pyplot

import sqlite3 as sql
import sanetime

import itertools

# Connect to database, load query into dataframe
DB_NAME="superuser.sqlite3"
con = sql.connect(DB_NAME)

# Load features as dataframes
usersToQuestionsFileName='usersToQuestions.npz'

# Convert CreationDate column from string into unix epoch timestamp
# (integer seconds since 1970)
def loadDataframe(queryString):
  dataframe = pd_sql.read_frame(queryString, con)
  #dataframe['CreationDate'] = dataframe['CreationDate'].apply(lambda t: sanetime.time(t).seconds)
  return dataframe

print 'Loading Users Dataframe'
numUsers=10000
# Users = contributors to stackoverflow
# - Id 
# - Reputation (int64) = total points received for participating in the community
# - CreatedDate (datetime) = date when the user joined superuser.com
users = loadDataframe("Select Id, Reputation, CreationDate From Users order by Reputation desc limit "+str(numUsers))


print 'Loading Questions Dataframe'
# Questions = 
# - Id
# - AcceptedId (id) = id of the answer that was accepted by the creator of this question (post.acceptedanswerid)
# - OwnerId (id) = id of the user who created the answer (post.owneruserid; -1 for wiki community answer)
# - CreationDate (datetime) = iso timestamp of when answer was created
# - Score (int64) - sum of up/downvotes that this question has received
# - FavoriteCount (int64) - number of users who have selected this as a favorite question?
# - Title (string) - only seems to be available for questions
# - Tags (series of string) - list/series of tag strings
questions = loadDataframe("Select Id as QuestionId, AcceptedAnswerId as AnswerId, OwnerUserId as OwnerId, CreationDate, Score, FavoriteCount, Title, Tags from Posts where PostTypeId=1 and Id in (Select ParentId from Posts where PostTypeId=2)")
numQuestions = len(questions)


# Tags is DataFrame containing:
# - Id = id of question this tag is associated with
# - OwnerId = id of user who asked question containing this tag
# - Tag - string representation of the tag.
# Note that a specific Tag can appear in multiple questions, but (Id, Tag) pairs are unique.
tags = questions[['QuestionId', 'OwnerId', 'Tags']]
# Replace u'<windows><disk-space><winsxs>' with pandas series [u'windows', u'disk-space', u'winsxs']
tagsColumn = tags['Tags'].apply(lambda tagString: pd.Series(tagString.strip("<>").split("><"))).stack()
# Reduce dimensionality of tags column: convert from column containing tuples to column with single words
# http://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-columns
tagsColumn.index = tagsColumn.index.droplevel(-1)
tagsColumn.name = 'Tags'
del tags['Tags']
tags = tags.join(tagsColumn)
# tags.reset_index(drop=True) #this doesn't seem to work...
tags.index=range(len(tags))


print 'Grouping Questions by Tag'
# Group by tag to determine relative frequencies
# http://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-object-to-dataframe
# http://stackoverflow.com/questions/18927238/how-to-split-a-pandas-dataframe-into-many-columns-after-groupby
# TagCounts is a DataFrame containing:
# - NumQuestions = number of questions labelled with this tag
# - NumAskers = number of users who asked a question containing this tag
tagCounts = tags.groupby('Tags').count()
tagCounts = tagCounts.rename(columns={'QuestionId':'NumQuestions', 'OwnerId':'NumAskers'})
del tagCounts['Tags']

# tagPriors captures probability that a tag appears in a question
# Computed as (num questions with this tag)/(total num questions)
totalNumQuestions=len(questions)
tagPriors = pd.DataFrame(data=tagCounts['NumQuestions'], columns=['Probability'], dtype='float32')

tagPriors = tagPriors/totalNumQuestions
tagPriors['Index'] = np.arange(0, len(tagPriors))

# Array of tag index to probability tag appears in question which can be used in computations
#tagPriorsArray = tagPriors['Probability'].values[0]

# Dictionary which maps from tag to its index (for building sparse matrices)
tagToIndex=dict(row for row in tagPriors['Index'].iteritems())


print 'Grouping Tags by Question'
# Compute vector of tag weights for each tag in question
# m x n matrix where m=num rows, n=num available tags
# row index corresponds to index of the question
# column index corresponds to tag from tagCounts
# each entry in matrix is probability that tag (column) appears in
# the question (row), computed as:
# (tagInQuestion ? 1 : 0)*(1.0-(probability tag appears in any question))
# NaN = tag not in question,
# low value (near 0) = question has tag, but tag present in many other questions (ie: common tag)
# high value (near 1) = question has tag, only present in few other questions (ie: rare tag)

# sparse.csr_matrix = Compressed Sparse Row matrix: column indices
# for row i are stored in indices[indptr[i]:indptr[i+1]] and their
# corresponding values are stored in data[indptr[i]:indptr[i+1]]. 

#@profile
def getQuestionsToTags():
  keywordIndexes = list()
  keywordProbabilities = list()
  questionIndexes = list()
  questionIndex=0
  # iterrows is really slow... https://groups.google.com/forum/#!topic/pystatsmodels/cfQOcrtOPlA
  for questionTags in questions['Tags']:
    # convert xml tags to list
    relevantTags = questionTags.strip("<>").split("><")
    questionTags = relevantTags
    # keep probabilities only for the available tags
    for tag in relevantTags:
      (probability,index)=tagPriors.loc[tag]
      # Note: this feature captures how rare a tag is
      keywordProbabilities.append(1.0-probability)
      keywordIndexes.append(int(index))
      questionIndexes.append(questionIndex)
    if questionIndex%10000 == 0:
      print questionIndex
    questionIndex+=1

  indexes = np.array((questionIndexes, keywordIndexes))
  return sparse.csr_matrix((keywordProbabilities, indexes), dtype='float32', shape=(len(questions), len(tagPriors)))

questionsToTags = getQuestionsToTags()


print 'Loading Answers Dataframe'
# Answers = 
# - id
# - questionid (id) = id of the question this answer is attached to (post.parentid)
# - ownerid (id) = id of the user who created the answer (-1 for wiki community answer)
# - creationdate (datetime) = iso timestamp of when answer was created
# - score (int64) - sum of up/downvotes that this answer has received
answers = loadDataframe("Select Id, ParentId as QuestionId, OwnerUserId as OwnerId, CreationDate, Score from Posts where PostTypeId=2 and OwnerUserId in (Select Id From Users Order by Reputation desc limit "+str(numUsers)+");")


print 'Grouping Tags by User'
# Build up UserToTag mappings, since that isn't included in data dump
def mergeAnswersTags(answersDataframe, tagsDataframe):
  tempTags = tags[['Tags', 'QuestionId']]
  tempAnswers = answers[['OwnerId','QuestionId']]
  tempAnswers=tempAnswers.rename(columns={'OwnerId':'Id'})
  # Step1: get all tags for each answer
  answersToTags=tempAnswers.merge(tempTags, on="QuestionId")
  # http://stackoverflow.com/questions/19530568/can-pandas-groupby-aggregate-into-a-list-rather-than-sum-mean-etc
  # Step 2: pivot/group tags by user, get number of times user has used that tag
  print 'Aggregating Tags by User'
  tagsGroupedByUser = answersToTags.groupby(['Id','Tags'])['QuestionId'].apply(lambda questionid: len(questionid.unique()))
  return tagsGroupedByUser

# Denormalized representation via multidimensional matrix;
# each row contains: answerer userId, tag, count.
usersToTagsMultidimensional=mergeAnswersTags(answers, tags)

# Sparse matrix representation: each row is a user, columns are tags
# elements are the number of times user used that tag
def getUserToTagsMatrix(usersToTagsMultidimensional):
  userIndex=0
  previousUserId=1
  tagIndexes = list()
  userIndexes = list()
  tagWeights = list()
  print 'Building sparse usersToTags matrix'
  for ((userid, tag), count) in usersToTagsMultidimensional.iteritems():
    if previousUserId != userid:
      # start new row
      userIndex += 1
      previousUserId=userid
    userIndexes.append(userIndex)
    tagIndexes.append(tagToIndex[tag])
    tagWeights.append(count)
  
  # Build sparse matrix
  indexes = np.array((userIndexes, tagIndexes))
  return sparse.csr_matrix((tagWeights, indexes), dtype='float32', shape=(len(users), len(tagPriors)))


# Normalize usersToTags sparse matrix so rows sum to 1
def getUsersToTagsSparse(usersToTagsMultidimensional):
  usersToTags = getUserToTagsMatrix(usersToTagsMultidimensional)
  rowSums = np.array(usersToTags.sum(axis=1))[:,0]
  rowIndices, colIndices = usersToTags.nonzero()
  usersToTags.data /= rowSums[rowIndices]
  del rowIndices
  del colIndices
  del rowSums
  return usersToTags

# save dataframes before removing them
usersToTags = getUsersToTagsSparse(usersToTagsMultidimensional)
del usersToTagsMultidimensional
del tagToIndex

del tags
del tagCounts

# Verify that rows sum to 1
#np.sum(usersToTags[0].todense())

# Example: take dot product of 1st row of usersToTags and questionsToTags
#np.asscalar(usersToTags.getrow(0).dot(questionsToTags.getrow(0).T).todense())

# Create giant matrix of users' affinity to questions...
# this results in MemoryError...
#usersToQuestions = usersToTags * questionsToTags.T

# Save sparse usersToQuestions matrix to disk
# A csr_matrix has 3 data attributes that matter:
# .data
# .indices
# .indptr
# All are simple ndarrays, so numpy.save will work on them. Save the three
# arrays with numpy.save or numpy.savez, load them back with numpy.load, and
# then recreate the sparse matrix object with:
#  new_csr = csr_matrix((data, indices, indptr), shape=(M, N))
def saveCSRMatrix(matrix, compressed=True):
  print 'Saving CSRMatrix to disk as '+usersToQuestionsFileName
  if compressed:
    # Compressed: slower save, but uses less disk space (~1GB)
    np.savez_compressed(usersToQuestionsFileName, matrix.data, matrix.indices, matrix.indptr)
  else:
    # Uncompressed: Faster save, but takes up a lot of disk space (3.3GB)
    np.savez(usersToQuestionsFileName, matrix.data, matrix.indices, matrix.indptr)

def loadCSRMatrix(fileName):
  npz = np.load(fileName)
  return sparse.csr_matrix((npz['arr_0'], npz['arr_1'], npz['arr_2']), dtype='float32')
  
#saveCSRMatrix(usersToQuestions)


# For a given question, which users are most likely to answer it,
# given the tags in that question?

#print 'Predicting users most likely to answer question'
#numTop=100
#questionIndex=0
#numHits=0
#for questionToTags in questionsToTags:
#  relevantUsers = usersToTags*questionToTags.T
#  #topUsers = bottleneck.argpartsort(-relevantUsers.toarray(), numTop, axis=0)
#  topUserIndexes = np.argsort(-relevantUsers.toarray(), axis=0)[0:numTop]
#  # Determine if user from topUsers answered the question
#  topUserIds = users['Id'].ix[topUserIndexes.flatten()]
#  questionId=questions['QuestionId'].ix[questionIndex]
#  results = answers[(answers['OwnerId'].isin(topUserIds)) & (answers['QuestionId']==questionId)]
#  if len(results) > 0:
#    numHits+=1
#  if questionIndex % 10000 == 0:
#    print questionIndex
#  questionIndex += 1
#
#print 'Prediction rate:'+str(numHits/float(len(questionsToTags)))

# For a given user, which questions is he most likely to answer?
print 'Predicting questions most likely answered by user'


# Build hashmap of questionId, answererId combinations
questionsOwnersSet=set(zip(answers.QuestionId,answers.OwnerId))

#@profile
def predictQuestionsAnsweredByUser():
  userIndex=0
  histogram = np.zeros(numQuestions, dtype='int32')
  for userToTags in usersToTags:
    relevantQuestions = questionsToTags*userToTags.T
    topQuestionIndexes = np.argsort(-relevantQuestions.toarray(), axis=0)
    topQuestions = questions.iloc[topQuestionIndexes.flatten()]
    userId = users['Id'].iloc[0]
    rankIndex = 0
    for questionId in topQuestions['QuestionId']:
      if (questionId, userId) in questionsOwnersSet:
        histogram[rankIndex] += 1
        break
      rankIndex += 1
    if userIndex % 1000 == 0:
      print userIndex
    userIndex += 1
  return histogram

hitHistogram = predictQuestionsAnsweredByUser()
# DO THIS:
np.savez('histogram', hitHistogram)

def plotHistogram(histogram):
  pyplot.Figure()
  pyplot.loglog(np.arange(0,len(histogram)), histogram, 'b.')
  #pyplot.legend(("Random Network Failure","Random Network Attack"),loc="best")
  pyplot.title('Ranks Of Questions That Users Answer')
  pyplot.xlabel("Question's rank on personalized recommendation list for user")
  pyplot.ylabel("Number of questions")
  pyplot.show(block=True)

plotHistogram(hitHistogram)
