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
import sqlite3 as sql

import itertools

# Connect to database, load query into dataframe
DB_NAME="superuser.sqlite3"
con = sql.connect(DB_NAME)

# Load features as dataframes

print 'Loading Users Dataframe'
# Users = contributors to stackoverflow
# - Id 
# - Reputation (int64) = total points received for participating in the community
# - CreatedDate (datetime) = date when the user joined superuser.com
users = pd_sql.read_frame("Select Id, Reputation, CreationDate From Users", con)


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
questions = pd_sql.read_frame("Select Id, AcceptedAnswerId as AnswerId, OwnerUserId as OwnerId, CreationDate, Score, FavoriteCount, Title, Tags from Posts where PostTypeId=1", con)

tags = questions[['Id', 'OwnerId', 'Tags']]
# Replace u'<windows><disk-space><winsxs>' with pandas series [u'windows', u'disk-space', u'winsxs']
tagsColumn = tags['Tags'].apply(lambda tagString: pd.Series(tagString.strip("<>").split("><"))).stack()
# Reduce dimensionality of tags column: convert from column containing tuples to column with single words
# http://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-columns
tagsColumn.index = tagsColumn.index.droplevel(-1)
tagsColumn.name = 'Tags'
del tags['Tags']
tags = tags.join(tagsColumn)


print 'Grouping Questions by Tag'
# Group by tag to determine relative frequencies
# http://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-object-to-dataframe
# http://stackoverflow.com/questions/18927238/how-to-split-a-pandas-dataframe-into-many-columns-after-groupby
tagCounts = tags.groupby('Tags').count()
totalNumTags = float(sum(tagCounts['Id']))
tagPriors = pd.DataFrame(data=tagCounts['Id'], columns=['Probability'])
tagPriors = tagPriors/totalNumTags
tagPriors['Index'] = np.arange(0, len(tagPriors))

# Array of tag index to probability which can be used in computations
tagPriorsArray = tagPriors['Probability'].values[0]


print 'Grouping Tags by Question'
# Compute vector of tag weights for each tag in question
# m x n matrix where m=num rows, n=num available tags
# row index corresponds to Id from questions
# column index corresponds to tag from tagCounts
# each entry in matrix is probability tag appears in question

# This results in MemoryError, need sparse matrix representation...
# questionsToTags = tags.pivot(index='Id', columns='Tags')
# Constructing sparse dataframes: https://groups.google.com/forum/#!msg/pydata/9c1vem2ksPQ/4F0X8nH5-HgJ
# questionsToTags = pd.SparseDataFrame???

# sparse.csr_matrix = Compressed Sparse Row matrix: column indices
# for row i are stored in indices[indptr[i]:indptr[i+1]] and their
# corresponding values are stored in data[indptr[i]:indptr[i+1]]. 

#@profile
def getQuestionsToTags():
  keywordIndexes = pd.Series()
  keywordProbabilities = pd.Series()
  questionIndexes = list()
  questionIndex=0
  # iterrows is really slow... https://groups.google.com/forum/#!topic/pystatsmodels/cfQOcrtOPlA
  for questionTags in questions['Tags']:
    # convert xml tags to list
    relevantTags = questionTags.strip("<>").split("><")
    questionTags = relevantTags
    # keep probabilities only for the available tags
    tagVector = tagPriors.loc[relevantTags]
    keywordProbabilities=keywordProbabilities.append(tagVector['Probability'])
    keywordIndexes=keywordIndexes.append(tagVector['Index'])
    questionIndexes=itertools.chain(questionIndexes, [questionIndex]*len(tagVector))
    if questionIndex%10000 == 0:
      print questionIndex
    questionIndex+=1

    
  print 'Building sparse questionsToTags matrix'
  indexes = np.vstack((np.array(list(questionIndexes)), keywordIndexes.values))
  return sparse.csr_matrix((keywordProbabilities, indexes), shape=(len(questions), len(tagPriors)))

questionsToTags = getQuestionsToTags()

print 'Grouping Tags by User - TODO'


print 'Loading Answers Dataframe'
# Answers = 
# - id
# - questionid (id) = id of the question this answer is attached to (post.parentid)
# - ownerid (id) = id of the user who created the answer (-1 for wiki community answer)
# - creationdate (datetime) = iso timestamp of when answer was created
# - score (int64) - sum of up/downvotes that this answer has received
answers = pd_sql.read_frame("Select Id, ParentId as QuestionId, OwnerUserId as OwnerId, CreationDate, Score from Posts where PostTypeId=2", con)


# Build up UserToTag and PostToTag mappings, since that isn't included in data dump
# This doesn't work since stackoverflow tags don't have closing slash; I hate xml...
# ElementTree.fromstring(tags)

