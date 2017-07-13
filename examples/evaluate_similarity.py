# -*- coding: utf-8 -*-

"""
 Simple example showing evaluating embedding on similarity datasets
"""
import logging
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999
from web.embeddings import fetch_GloVe
from web.evaluate import evaluate_similarity

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch GloVe embedding (warning: it might take few minutes)
w_glove = fetch_GloVe(corpus="wiki-6B", dim=300)

# Define tasks
tasks = {
    "MEN": fetch_MEN(),
    "WS353": fetch_WS353(),
    "SIMLEX999": fetch_SimLex999()
}

# Print sample data
for name, data in iteritems(tasks):
    print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data.X[0][0], data.X[0][1], data.y[0]))

# Calculate results using helper function
for name, data in iteritems(tasks):
    print "Spearman correlation of scores on {} {}".format(name, evaluate_similarity(w_glove, data.X, data.y))

