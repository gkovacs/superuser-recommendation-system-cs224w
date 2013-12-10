# compute communities based on user tag-usage similarity

#from features import *

import sys
import math
import sanetime

#import snap as snap
import networkx as nx
import community

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
import cPickle

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
  #for (userIndex, userId) in users['Id'].iteritems():
  for (userIndex, userId) in itertools.islice(users['Id'].iteritems(), 1000, 2000):
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
cPickle.dump(userGraph, f)
f.close()

#print 'Loading graph from disk'
#f = open('userGraph')
#userGraph = cPickle.load(f)
#first compute the best partition
print 'Computing partitions'
partitions = community.best_partition(userGraph)
import json
with open('partitions.txt', 'w') as outfile:
  json.dump(partitions, outfile)

partitionsCounter = Counter(partition.values())
import ipdb
ipdb.set_trace()


def drawPartitions():
  print 'Drawing partitions'
  # draw top 5 partitions
  #blue, green, red, cyan, magenta, yellow
  colors = ['b', 'g', 'r', 'c', 'm'] #'y'
  size = float(len(colors)) #float(len(set(partitions.values())))
  pos = nx.spring_layout(userGraph)
  partitionsCounter = Counter(partitions.values())
  relevantPartitions = {partitionKey: partitionValue for (partitionKey, partitionValue)
    in partitionsCounter.items() if partitionValue > 100}
  for partitionKey in relevantPartitions.keys():
    list_nodes = [nodes for nodes in partitions.keys()
      if partitions[nodes] == partitionKey]
    nx.draw_networkx_nodes(userGraph, pos, list_nodes, node_size = 20,
      node_color = colors[partitionKey])
  
  nx.draw_networkx_edges(userGraph, pos, alpha=0.5)
  plt.savefig('partitions.png')


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
