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

from collections import Counter, defaultdict
import itertools
import cPickle
import operator
import json

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
userIndexToId=dict(userIndexId for userIndexId in users['Id'].iteritems())

def loadCSRMatrix(fileName):
  npz = np.load(fileName)
  return sparse.csr_matrix((npz['arr_0'], npz['arr_1'], npz['arr_2']), dtype='float32')

print 'Loading usersToTags CSRMatrix'
usersToTags = loadCSRMatrix(USERS_TO_TAGS)

#@profile
def buildGraph():
  TOPK=1000 # number of edges to keep
  adjacencyMatrix = np.zeros((numUsers,numUsers), dtype='float32')
  for (userIndex, userId) in users['Id'].iteritems():
  #for (userIndex, userId) in itertools.islice(users['Id'].iteritems(), 1000, 2000):
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

from os.path import exists
from sys import exit
if not exists('userGraph'):
  userGraph = buildGraph()
  del usersToTags
  del users
  print 'Writing graph to disk'
  f = open('userGraph', 'w')
  cPickle.dump(userGraph, f)
  f.close()
  print 'generated file userGraph, now exiting, please restart script'
  exit()

#print 'Loading graph from disk'
f = open('userGraph')
userGraph = cPickle.load(f)

if not exists('partitions.txt'):
  #first compute the best partition
  print 'Computing partitions'
  partitions = community.best_partition(userGraph)

  with open('partitions.txt', 'w') as outfile:
    json.dump(partitions, outfile)
  print 'generated file partitions.txt, now exiting, please restart script'
  exit()

with open('partitions.txt') as infile:
  partitions = json.load(infile)

# Get list of userIds in each partition:
partitionsToUsers = defaultdict(list)
for userIndex, partition in partitions.iteritems():
  partitionsToUsers[partition].append(userIndexToId[int(userIndex)])

relevantPartitions = {partitionKey: len(partitionValues) for (partitionKey, partitionValues)
  in partitionsToUsers.items() if len(partitionValues) > 100}

def getTags():
  # Determine which tags are associated with each partition
  print 'Loading Tags Dataframe'
  
  # Tags is DataFrame containing:
  # - Id = id of question this tag is associated with
  # - OwnerId = id of user who asked question containing this tag
  # - Tag - string representation of the tag.
  # Note that a specific Tag can appear in multiple questions, but (Id, Tag) pairs are unique.
  tags = loadDataframe("Select Id as QuestionId, OwnerUserId as OwnerId, Tags from Posts where PostTypeId=1 and Id in (Select ParentId from Posts where PostTypeId=2)")
  
  # Replace u'<windows><disk-space><winsxs>' with pandas series [u'windows', u'disk-space', u'winsxs']
  tagsColumn = tags['Tags'].apply(lambda tagString: pd.Series(tagString.strip("<>").split("><"))).stack()
  # Reduce dimensionality of tags column: convert from column containing tuples to column with single words
  # http://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-columns
  tagsColumn.index = tagsColumn.index.droplevel(-1)
  tagsColumn.name = 'Tags'
  del tags['Tags']
  tags = tags.join(tagsColumn)
  tags.index=range(len(tags))
  return tags

tags = getTags()
print 'Grouping keywords by partition'
for partitionKey, userCounts in relevantPartitions.items():
  tagsForPartition = tags['Tags'][np.in1d(tags['OwnerId'], partitionsToUsers[partitionKey])]
  partitionFileName = 'partition'+str(partitionKey)+'.json'
  tagsForPartition.value_counts().to_json(partitionFileName)

def drawPartitions():
  print 'Drawing partitions'
  # draw top 6 partitions
  #blue, green, red, cyan, magenta, yellow
  colors = ['b', 'g', 'r', 'c', 'm', 'y']
  pos = nx.spring_layout(userGraph)
  for partitionKey in relevantPartitions.keys():
    list_nodes = [nodes for nodes in partitions.keys()
      if partitions[nodes] == partitionKey]
    nx.draw_networkx_nodes(userGraph, pos, list_nodes, node_size = 20,
      node_color = colors[partitionKey])
  
  nx.draw_networkx_edges(userGraph, pos, alpha=0.5)
  plt.savefig('partitions.png')

drawPartitions()

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
