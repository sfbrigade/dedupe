#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Adapted version of csv_example.py that allows for automated training from the
true results. Specify a %age at the command line for how much training to do.
E.g.

 csv_auto_example.py 5

To train by answering questions for 5% of the total number of records.

"""

import os
import sys
import csv
import re
import collections
import logging
import optparse

import AsciiDammit

import dedupe

# ## Logging

# Dedupe uses Python logging to show or suppress verbose output. Added for convenience.
# To enable verbose logging, run `python examples/csv_example/csv_example.py -v`

optp = optparse.OptionParser()
optp.add_option('-v', '--verbose', dest='verbose', action='count',
                help='Increase verbosity (specify multiple times for more)'
                )
optp.add_option('-q', '--quota', dest='quota', type='int',
                help='How many training questions to answer as %%age of size of data')
(opts, args) = optp.parse_args()
log_level = logging.WARNING 
if opts.verbose == 1:
    log_level = logging.INFO
elif opts.verbose >= 2:
    log_level = logging.DEBUG
logging.basicConfig(level=log_level)
if not opts.quota:
    print "You must supply a quota. E.g. -q5"
    sys.exit()

# ## Setup

# Switch to our working directory and set up our input and out put paths,
# as well as our settings and training file locations
os.chdir('./examples/csv_example/')
input_file = 'csv_example_messy_input.csv'
output_file = 'csv_example_output.csv'
settings_file = 'csv_example_learned_settings'
training_file = 'csv_example_training.json'
true_clusters_file = 'csv_example_input_with_true_ids.csv'


# Dedupe can take custom field comparison functions, here's one
# we'll use for zipcodes
def sameOrNotComparator(field_1, field_2) :
    if field_1 == field_2 :
        return 1
    else:
        return 0



def preProcess(column):
    """
    Do a little bit of data cleaning with the help of [AsciiDammit](https://github.com/tnajdek/ASCII--Dammit) 
    and Regex. Things like casing, extra spaces, quotes and new lines can be ignored.
    """

    column = AsciiDammit.asciiDammit(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    return column


def readData(filename):
    """
    Read in our data from a CSV file and create a dictionary of records, 
    where the key is a unique record ID and each value is a 
    [frozendict](http://code.activestate.com/recipes/414283-frozen-dictionaries/) 
    (hashable dictionary) of the row fields.

    **Currently, dedupe depends upon records' unique ids being integers
    with no integers skipped. The smallest valued unique id must be 0 or
    1. Expect this requirement will likely be relaxed in the future.**
    """

    data_d = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            row_id = int(row['Id'])
            data_d[row_id] = dedupe.core.frozendict(clean_row)

    return data_d


print 'importing data ...'
data_d = readData(input_file)

### Make a label function that uses the true results...
true_clusters_data = readData(true_clusters_file)
true_id_to_cluster = {}
for id, data in true_clusters_data.iteritems():
    true_id_to_cluster[id] = data['True Id']


num_trained = 0
sample_size = len(data_d)
target_trained = (opts.quota*sample_size)/100

print "Training quota is %s%%, so that's %s training questions." % (opts.quota,
                    target_trained)

def rainman(record_pairs, data_model) :
    # rainman is a labelling function that is always right.
    
    global num_trained, sample_size, target_trained, true_id_to_cluster

    finished = False
    duplicates = []
    nonduplicates = []

    for record_pair in record_pairs:
        id_1 = int(record_pair[0]['Id'])
        id_2 = int(record_pair[1]['Id'])

        if true_id_to_cluster[id_1] == true_id_to_cluster[id_2]:
            duplicates.append(record_pair)
        else:
            nonduplicates.append(record_pair)

        num_trained += 1
        if num_trained >= target_trained:
            finished = True
            break

    return ({0: nonduplicates, 1: duplicates}, finished)

# ## Training

if True:
    # To train dedupe, we feed it a random sample of records.
    data_sample = dedupe.dataSample(data_d, 150000)

    # Define the fields dedupe will pay attention to
    #
    # Notice how we are telling dedupe to use a custom field comparator
    # for the 'Zip' field. 
    fields = {
        'Site name': {'type': 'String'},
        'Address': {'type': 'String'},
        'Zip': {'type': 'String', 'Has Missing':True},
        'Phone': {'type': 'String', 'Has Missing':True},
        }

    # Create a new deduper object and pass our data model to it.
    deduper = dedupe.Dedupe(fields)

    # ## Active learning

    # Starts the training loop. Dedupe will find the next pair of records
    # it is least certain about and ask you to label them as duplicates
    # or not.

    # use 'y', 'n' and 'u' keys to flag duplicates
    # press 'f' when you are finished
    print 'starting active labeling...'
    deduper.train(data_sample, rainman)

    # When finished, save our training away to disk
    deduper.writeTraining(training_file)

# ## Blocking

print 'blocking...'
# Initialize our blocker. We'll learn our blocking rules if we haven't
# loaded them from a saved settings file.
blocker = deduper.blockingFunction()

# Save our weights and predicates to disk.  If the settings file
# exists, we will skip all the training and learning next time we run
# this file.
deduper.writeSettings(settings_file)

# Load all the original data in to memory and place
# them in to blocks. Each record can be blocked in many ways, so for
# larger data, memory will be a limiting factor.

blocked_data = dedupe.blockData(data_d, blocker)

# ## Clustering

# Find the threshold that will maximize a weighted average of our precision and recall. 
# When we set the recall weight to 2, we are saying we care twice as much
# about recall as we do precision.
#
# If we had more data, we would not pass in all the blocked data into
# this function but a representative sample.

threshold = deduper.goodThreshold(blocked_data, recall_weight=2)

# `duplicateClusters` will return sets of record IDs that dedupe
# believes are all referring to the same entity.

print 'clustering...'
clustered_dupes = deduper.duplicateClusters(blocked_data, threshold)

print '# duplicate sets', len(clustered_dupes)

# ## Writing Results

# Write our original data back out to a CSV with a new column called 
# 'Cluster ID' which indicates which records refer to each other.

cluster_membership = collections.defaultdict(lambda : 'x')
for (cluster_id, cluster) in enumerate(clustered_dupes):
    for record_id in cluster:
        cluster_membership[record_id] = cluster_id


with open(output_file, 'w') as f:
    writer = csv.writer(f)

    with open(input_file) as f_input :
        reader = csv.reader(f_input)

        heading_row = reader.next()
        heading_row.insert(0, 'Cluster ID')
        writer.writerow(heading_row)

        for row in reader:
            row_id = int(row[0])
            cluster_id = cluster_membership[row_id]
            row.insert(0, cluster_id)
            writer.writerow(row)

print "Now run csv_auto_example.py to check how well it did."
