import matplotlib.pyplot as pyplot
import json
import operator
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

partitions=OrderedDict()
for partition in range(6):
  name = 'partition'+str(partition)
  with file(name+'.json') as infile:
    partitionCounter = Counter(json.load(infile))
    partitions[name]=partitionCounter

allTagCounts = reduce(operator.add, partitions.values(), Counter())

# sort alphabetically
allTags, allValues = zip(*sorted(allTagCounts.items(), key=operator.itemgetter(0)))
allTagFrequencies = pd.Series(allValues, index=allTags)
# normalize by number of tags
allTagFrequencies = allTagFrequencies/float(sum(allTagFrequencies))


frequenciesDict=OrderedDict()
for partition in partitions.keys():
  tags, values = zip(*sorted(partitions[partition].items(), key=operator.itemgetter(0)))
  values = np.array(values)
  values = values/float(np.sum(values))
  frequenciesDict[partition] = pd.Series(values, index=tags)

import ipdb
ipdb.set_trace()
frequencies=pd.DataFrame(frequenciesDict)
del frequenciesDict

tags = tags[0:100]
values = values[0:100]

#pyplot.bar(range(len(values)), values, align='center')
pyplot.bar(range(len(values)), values)
pyplot.xticks(range(len(values)), tags, rotation=90)
pyplot.show()
