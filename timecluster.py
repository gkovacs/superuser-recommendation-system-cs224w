#finding P(Q,T)

from features import *

import sys
import math
import sanetime


question_dict = dict()

print 'building question_dict'

for i in questions.index:
	time_in_sec = sanetime.time(questions.ix[i][3]).seconds
	#time since 1970 in seconds. 
	if math.isnan(questions.ix[i][1]): #question is not sufficiently answered.
		question_dict[questions.ix[i][0]] = (False, time_in_sec)
	else:
		question_dict[questions.ix[i][0]] = (True, time_in_sec)
	#add (answerID, time) pair to dictionary for O(1) lookup. 

print 'buildling time_delta_ans'

#populate the deltas (question answered time - question asked time in seconds.)
time_delta_ans = []
time_delta_unans = []

for i in answers.index:
	question_a = question_dict[answers.ix[i][1]][0] #question is sufficiently answered.
	question_t = question_dict[answers.ix[i][1]][1] #time question was asked.
	answered_t = sanetime.time(answers.ix[i][3]).seconds #time answered.
	delta = answered_t - question_t
	if question_a:
		time_delta_ans.append(delta)
	else:
		time_delta_unans.append(delta)


def bucketList(time_delta, num_buckets):
	time_min = sys.maxint
	time_max = -1
	for delta in time_delta:
		if delta > time_max:
			time_max = delta
		if delta < time_min:
			time_min = delta

	spread = time_max - time_min + 2 
	#lower bound is time_min-1 and upper is time_max+1

	bucket_s = spread / num_buckets

	if spread % num_buckets != 0: #last bucket is left out because of int division.
		bucket_s += 1

	numAnswers = len(time_delta) #equivalent to numAnswered since each answer has time delta.

	prob_vec = [0 for i in range(num_buckets)]

	for delta in time_delta:
		bucket_index = delta / bucket_s
		prob_vec[bucket_index] += 1.0 / numAnswers #add fraction of occurences.

	return (bucket_s, prob_vec)

print 'running bucketList'

#we need these to compute P(Q,T) for other questions
(buck_s_sec,prob_ans) = bucketList(time_delta_ans, 1000)
(buck_s_sec,prob_unans) = bucketList(time_delta_unans, 1000)


print buck_s_sec
print prob_ans



