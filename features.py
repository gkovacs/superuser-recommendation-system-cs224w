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
import sqlite3 as sql

# Connect to database, load query into dataframe
DB_NAME="superuser.sqlite3"
con = sql.connect(DB_NAME)

# Load features as dataframes

# Users = contributors to stackoverflow
# - Id 
# - Reputation (int64) = total points received for participating in the community
# - CreatedDate (datetime) = date when the user joined superuser.com
users = pd_sql.read_frame("Select Id, Reputation, CreationDate From Users", con)


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
# Replace u'<windows><disk-space><winsxs>' with pandas series [u'windows', u'disk-space', u'winsxs']
tagsColumn = questions['Tags'].apply(lambda tags: pd.Series(tags.strip("<>").split("><"))).stack()
# Reduce dimensionality of tags column: convert from column containing tuples to column with single words
# http://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-columns
tagsColumn.index = tagsColumn.index.droplevel(-1)
tagsColumn.name = 'Tags'
del questions['Tags']
questions = questions.join(tagsColumn)


# Group by tag to determine relative frequencies
# http://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-object-to-dataframe
# http://stackoverflow.com/questions/18927238/how-to-split-a-pandas-dataframe-into-many-columns-after-groupby

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

