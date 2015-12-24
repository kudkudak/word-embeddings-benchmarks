#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 This script calculates embedding results against all available fast running
 benchmarks in the repository and saves results as single row csv table.

 Usage: ./evaluate_on_all -f <path to file> -o <path to output file>

 NOTE:
 * script doesn't evaluate on WordRep (nor its subset) as it is non standard
 for now and long running (unless some nearest neighbor approximation is used).

 * script is using CosAdd for calculating analogy answer.

 * script is not reporting results per category (for instance semantic/syntactic) in analogy benchmarks.
 It is easy to change it by passing category parameter to evaluate_analogy function (see help).
"""
from optparse import OptionParser
import logging
import pandas as pd
import os
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW
from web.datasets.analogy import fetch_google_analogy, fetch_msr_analogy
from web.datasets.categorization import fetch_AP, fetch_battig, fetch_BLESS, fetch_ESSLI_1a, fetch_ESSLI_2b, \
    fetch_ESSLI_2c
from web.embeddings import fetch_GloVe, load_embedding
from web.similarity import evaluate_similarity
from web.categorization import evaluate_categorization
from web.analogy import evaluate_analogy, evaluate_on_semeval_2012_2
from web.datasets.utils import _get_dataset_dir

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)


parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="Path to the file with embedding. If relative will load from data directory.",
                  default=None)

parser.add_option("-p", "--format", dest="format",
                  help="Format of the embedding, possible values are: word2vec, word2vec_bin, dict and glove.",
                  default=None)

parser.add_option("-o", "--output", dest="output",
                  help="Path where to save results.",
                  default=None)


parser.add_option("-c", "--clean_words", dest="clean_words",
                  help="Clean_words argument passed to load_embedding function. If set to True will remove"
                       "most of the non-alphanumeric characters, which should speed up evaluation.",
                  default=False)

if __name__ == "__main__":
    (options, args) = parser.parse_args()

    # Load embeddings
    fname = options.filename
    if not fname:
        w = fetch_GloVe(corpus="wiki-6B", dim=300)
    else:
        if not os.path.isabs(fname):
            fname = os.path.join(_get_dataset_dir(), fname)

        format = options.format

        if not format:
            _, ext = os.path.splitext(fname)
            if ext == ".bin":
                format = "word2vec_bin"
            elif ext == ".txt":
                format = "word2vec"
            elif ext == ".pkl":
                format = "dict"

        assert format in ['word2vec_bin', 'word2vec', 'glove', 'bin'], "Unrecognized format"

        load_kwargs = {}
        if format == "glove":
            vocab_size = sum(1 for line in open(fname))
            dim = len(next(open(fname)).split()) - 1

        w = load_embedding(fname, format=format, normalize=True, lower=True, clean_words=options.clean_words,
                           load_kwargs=load_kwargs)

    out_fname = options.output if options.output else "results.csv"

    # Calculate results on similarity
    logger.info("Calculating similarity benchmarks")
    similarity_tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "WS353R": fetch_WS353(which="relatedness"),
        "WS353S": fetch_WS353(which="similarity"),
        "SimLex999": fetch_SimLex999(),
        "RW": fetch_RW(),
        "RG65": fetch_RG65(),
        "MTurk": fetch_MTurk(),
    }

    similarity_results = {}

    for name, data in iteritems(similarity_tasks):
        similarity_results[name] = evaluate_similarity(w, data.X, data.y)
        logger.info("Spearman correlation of scores on {} {}".format(name, similarity_results[name]))

    # Calculate results on analogy
    logger.info("Calculating analogy benchmarks")
    analogy_tasks = {
        "Google": fetch_google_analogy(),
        "MSR": fetch_msr_analogy()
    }

    analogy_results = {}

    for name, data in iteritems(analogy_tasks):
        analogy_results[name] = evaluate_analogy(w, data.X, data.y)
        logger.info("Analogy prediction accuracy on {} {}".format(name, analogy_results[name]))


    analogy_results["SemEval2012_2"] = evaluate_on_semeval_2012_2(w)['all']
    logger.info("Analogy prediction accuracy on {} {}".format("SemEval2012", analogy_results["SemEval2012_2"]))

    # Calculate results on categorization
    logger.info("Calculating categorization benchmarks")
    categorization_tasks = {
        "AP": fetch_AP(),
        "BLESS": fetch_BLESS(),
        "Battig": fetch_battig(),
        "ESSLI_2c": fetch_ESSLI_2c(),
        "ESSLI_2b": fetch_ESSLI_2b(),
        "ESSLI_1a": fetch_ESSLI_1a()
    }

    categorization_results = {}

    # Calculate results using helper function
    for name, data in iteritems(categorization_tasks):
        categorization_results[name] = evaluate_categorization(w, data.X, data.y)
        logger.info("Cluster purity on {} {}".format(name,  categorization_results[name]))

    # Save csv
    cat = pd.DataFrame([categorization_results], index=["glove"])
    analogy = pd.DataFrame([analogy_results], index=["glove"])
    sim = pd.DataFrame([similarity_results], index=["glove"])
    results = cat.join(sim).join(analogy)

    logger.info("Saving results...")
    print(results)
    results.to_csv(out_fname)
