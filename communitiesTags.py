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

frequencies=pd.DataFrame(frequenciesDict)
del frequenciesDict

# find tags which are not present in a partition:
#frequencies['partition0'].index[frequencies['partition0'].apply(np.isnan)]
# fill missing values with 0
frequencies = frequencies.fillna(value=0)

frequenciesDiff = frequencies.sub(allTagFrequencies, axis=0)

for partition in partitions.keys():
  relativeFrequencies = dict(frequenciesDiff.sort([partition], ascending=0)[partition].iteritems())
  with open(partition+'-freqs.json', 'w') as outfile:
    json.dump(relativeFrequencies, outfile)

