# compute communities based on user tag-usage similarity

#from features import *

import sys
import math
import sanetime

#import snap as snap
import networkx as nx

import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import pandas.io.sql as pd_sql
from scipy import sparse
import sqlite3 as sql

from collections import Counter
import itertools
import sPickle

DB_NAME="superuser.sqlite3"
USERS_TO_TAGS='usersToTags.npz'
con = sql.connect(DB_NAME)

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

def loadCSRMatrix(fileName):
  npz = np.load(fileName)
  return sparse.csr_matrix((npz['arr_0'], npz['arr_1'], npz['arr_2']), dtype='float32')

print 'loading usersToTags CSRMatrix'

# graph where nodeId is index of user in users dataframe, userId attribute is stackoverflow UserId
#userGraph = nx.Graph()
#for (userIndex, userId) in users['Id'].iteritems():
#  userGraph.add_node(userIndex, userId=userId)

#userGraph = snap.TUNGraph.New()

usersToTags = loadCSRMatrix(USERS_TO_TAGS)
#import ipdb
#ipdb.set_trace()

#@profile
def buildGraph():
  TOPK=1000 # number of edges to keep
  adjacencyMatrix = np.zeros((numUsers,numUsers), dtype='float32')
  for (userIndex, userId) in users['Id'].iteritems():
  #for (userIndex, userId) in itertools.islice(users['Id'].iteritems(), 100):
    if userIndex % 100 == 0:
      print str(userIndex)+' of '+str(numUsers)
    #if userIndex == 10:
    #  from guppy import hpy; heaptest=hpy()
    #  heaptest.heap()
    #  import ipdb
    #  ipdb.set_trace()
    userSimilarities = usersToTags*usersToTags[userIndex].transpose()
    userSimilarities = userSimilarities.transpose().toarray()[0]
    # take top 1000
    threshold = np.sort(userSimilarities)[-TOPK]
    userSimilarities[userSimilarities<threshold]=0
    adjacencyMatrix[userIndex] = userSimilarities
  return nx.from_numpy_matrix(adjacencyMatrix)

userGraph = buildGraph()
del usersToTags
del users
print 'Writing graph to disk'
f = open('userGraph', 'w')
sPickle.s_dump(userGraph, f)
f.close()

#f = open('userGraph')
#userGraph = pickle.load(f)
#import ipdb
#ipdb.set_trace()

#@profile
#def getRanks():


#plt.xlabel('Ranks')
#plt.ylabel('Frequency')
#plt.title('Histogram of Ranks')
#plt.plot(Counter(ranks).keys(),Counter(ranks).values())
#plt.savefig('scores_time.png')
#
#import json
#with open('ranks.txt', 'w') as outfile:
#      json.dump(ranks, outfile)
#
#print ranks
