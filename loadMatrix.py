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

import sanetime

import itertools

usersToQuestionFileName='usersToQuestions.npz'

# Load sparse usersToQuestions matrix from disk
# A csr_matrix has 3 data attributes that matter:
# .data
# .indices
# .indptr
# All are simple ndarrays, so numpy.save will work on them. Save the three
# arrays with numpy.save or numpy.savez, load them back with numpy.load, and
# then recreate the sparse matrix object with:
#  new_csr = csr_matrix((data, indices, indptr), shape=(M, N))
def saveCSRMatrix(matrix):
  # Uncompressed: Faster save, but takes up a lot of disk space (3.3GB)
  np.savez(usersToQuestionFileName, matrix.data, matrix.indices, matrix.indptr)
  # Compressed: slower save, but uses less disk space (~1GB)
  #np.savez_compressed('usersToQuestions', matrix.data, matrix.indices, matrix.indptr)

def loadCSRMatrix(fileName):
  npz = np.load(fileName)
  return sparse.csr_matrix((npz['arr_0'], npz['arr_1'], npz['arr_2']), dtype='float32')

# Sparse m x n matrix where m=10,000 users with highest reputation,
#   n=160000 questions which have answers
# row index corresponds to index of the user
# column index corresponds to index (row) in questions dataframe
# usersToQuestions = usersToTags * questionsToTags.T

# each entry in matrix is proportional to that user (row) answering a
# particular question question (row), computed as:
# low value (near 0) = user did not answer questions with similar tags to this question, or tag is present in many other questions (ie: common tag)
# high value (near 1) = question has tag, only present in few other questions (ie: rare tag) which user answered

# sparse.csr_matrix = Compressed Sparse Row matrix: column indices
# for row i are stored in indices[indptr[i]:indptr[i+1]] and their
# corresponding values are stored in data[indptr[i]:indptr[i+1]]. 

# Sparse matrix representation: each row is a user, columns are tags
# elements are the number of times user used that tag
usersToQuestions = loadCSRMatrix(usersToQuestionFileName)

# Some examples:
# Verify that rows sum to 1
# np.sum(usersToTags[0].todense())

# Example: take dot product of 1st row of usersToTags and questionsToTags
#np.asscalar(usersToTags.getrow(0).dot(questionsToTags.getrow(0).T).todense())

# Create giant matrix of users' affinity to questions...
# this results in MemoryError...

