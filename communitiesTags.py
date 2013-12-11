import matplotlib.pyplot as pyplot
import json
import operator
from collections import Counter

import pandas as pd

partitions=[]
frequencies=[]
#frequencies=pd.DataFrame()
for partition in range(6):
  with file('partition'+str(partition)+'.json') as infile:
    partitionCounter = Counter(json.load(infile))
    tags, values = zip(*sorted(partitionCounter.items(), key=operator.itemgetter(0)))
    frequency = pd.Series(values, index=tags)
    frequencies.append(frequency/float(sum(frequency)))
    #frequencies['partition'+str(partition)]=frequency/sum(frequency))
    partitions.append(partitionCounter)

allTagCounts = reduce(operator.add, partitions, Counter())

# sort alphabetically
allTags, allValues = zip(*sorted(allTagCounts.items(), key=operator.itemgetter(0)))
allTagFrequencies = pd.Series(allValues, index=allTags)
# normalize by number of tags
allTagFrequencies = allTagFrequencies/float(sum(allTagFrequencies))

import ipdb
ipdb.set_trace()


tags = tags[0:100]
values = values[0:100]

#pyplot.bar(range(len(values)), values, align='center')
pyplot.bar(range(len(values)), values)
pyplot.xticks(range(len(values)), tags, rotation=90)
pyplot.show()
