# -*- coding: utf-8 -*-

"""
 Functions for fetching analogy datasets
"""

from collections import defaultdict
import glob
import os
import numpy as np

from sklearn.utils import check_random_state

from sklearn.datasets.base import Bunch
from .utils import _get_dataset_dir, _fetch_file, _change_list_to_np
from ..utils import standardize_string


def fetch_wordrep(subsample=None, rng=None):
    """
    Fetch MSR WordRep dataset for testing both syntactic and semantic dataset

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of word pairs
        'y': vector of answers
        'category': name of category
        'category_high_level': name of high level category (semantic/syntactic)

    References
    ----------
    Gao, Bin and Bian, Jiang and Liu, Tie-Yan,
    "Wordrep: A benchmark for research on learning word representations", 2014


    Notes
    -----
    This dataset is too big to calculate and store all word analogy quadruples, this is
    why it returns word paris

    """
    path = _fetch_file(url="https://www.dropbox.com/sh/5k78h9gllvc44vt/AAALLQq-Bge605OIMlmGBbNJa?dl=1",
                       data_dir="analogy",
                       uncompress=True,
                       move="EN-WORDREP/EN-WORDREP.zip",
                       verbose=0)

    wikipedia_dict = glob.glob(os.path.join(path, "Pairs_from_Wikipedia_and_Dictionary/*.txt"))
    wordnet = glob.glob(os.path.join(path, "Pairs_from_WordNet/*.txt"))

    # This dataset is too big to calculate and store all word analogy quadruples
    word_pairs = []
    category = []
    category_high_level = []

    files = wikipedia_dict + wordnet

    for file_name in files:
        c = os.path.basename(file_name).split(".")[0]
        c = c[c.index("-")+1:]
        with open(file_name, "r") as f:
            for l in f.read().splitlines():
                word_pairs.append(standardize_string(l).split())
                category.append(c)
                category_high_level.append("wikipedia-dict" if file_name in wikipedia_dict else "wordnet")

    if subsample:
        assert 0 <= subsample <= 1.0
        rng = check_random_state(rng)
        ids = rng.choice(range(len(word_pairs)), int(subsample * len(word_pairs)), replace=False)
        word_pairs = [word_pairs[i] for i in ids]
        category = [category[i] for i in ids]
        category_high_level = [category_high_level[i] for i in ids]

    wordnet_categories = {'Antonym',
     'Attribute',
     'Causes',
     'DerivedFrom',
     'Entails',
     'HasContext',
     'InstanceOf',
     'IsA',
     'MadeOf',
     'MemberOf',
     'PartOf',
     'RelatedTo',
     'SimilarTo'}

    wikipedia_categories = {'adjective-to-adverb',
     'all-capital-cities',
     'city-in-state',
     'comparative',
     'currency',
     'man-woman',
     'nationality-adjective',
     'past-tense',
     'plural-nouns',
     'plural-verbs',
     'present-participle',
     'superlative'}

    return Bunch(category_high_level=np.array(category_high_level),
                 X=np.array(word_pairs),
                 category=np.array(category),
                 wikipedia_categories=wordnet_categories,
                 wordnet_categories=wikipedia_categories)


def fetch_google_analogy():
    """
    Fetch Google dataset for testing both semantic and syntactic analogies.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of word questions
        'y': vector of answers
        'category': name of category
        'category_high_level': name of high level category (semantic/syntactic)

    References
    ----------
    Mikolov, Tomas and Sutskever, Ilya and Chen, Kai and Corrado, Greg S and Dean, Jeff,
    "Distributed representations of words and phrases and their compositionality", 2013

    Notes
    -----
    This dataset is a subset of WordRep dataset.

    """

    url = "https://www.dropbox.com/s/eujtyfb5zem1mim/EN-GOOGLE.txt?dl=1"
    with open(_fetch_file(url, "analogy/EN-GOOGLE", verbose=0), "r") as f:
        L = f.read().splitlines()

    # Simple 4 word analogy questions with categories
    questions = []
    answers = []
    category = []
    cat = None
    for l in L:
        if l.startswith(":"):
            cat =l.lower().split()[1]
        else:
            words =  standardize_string(l).split()
            questions.append(words[0:3])
            answers.append(words[3])
            category.append(cat)

    assert set(category) == set(['gram3-comparative', 'gram8-plural', 'capital-common-countries',
                                         'city-in-state', 'family', 'gram9-plural-verbs', 'gram2-opposite',
                                         'currency', 'gram4-superlative', 'gram6-nationality-adjective',
                                         'gram7-past-tense',
                                         'gram5-present-participle', 'capital-world', 'gram1-adjective-to-adverb'])


    syntactic = set([c for c in set(category) if c.startswith("gram")])
    category_high_level = []
    for cat in category:
         category_high_level.append("syntactic" if cat in syntactic else "semantic")

    # dtype=object for memory efficiency
    return Bunch(X=np.vstack(questions).astype("object"),
                 y=np.hstack(answers).astype("object"),
                 category=np.hstack(category).astype("object"),
                 category_high_level=np.hstack(category_high_level).astype("object"))



def fetch_msr_analogy():
    """
    Fetch MSR dataset for testing performance on syntactic analogies

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of word questions
        'y': vector of answers
        'category': name of category
        'category_high_level': name of high level category (noun/adjective/verb)

    References
    ----------
    Originally published at http://research.microsoft.com/en-us/projects/rnn/.

    Notes
    -----
    Authors description: "more precisely, we tagged 267M words of newspaper text
    with Treebank POS tags (Marcus et al., 1993). We then selected 100 of the most frequent comparative adjectives
    (words labeled JJR); 100 of the most frequent plural nouns (NNS); 100 of the most frequent possessive nouns
    (NN POS); and 100 of the most frequent base form verbs (VB).
    We then systematically generated analogy questions by randomly matching each of the 100 words with 5 other words
    from the same category, and creating variants.
    """
    url = "https://www.dropbox.com/s/ne0fib302jqbatw/EN-MSR.txt?dl=1"
    with open(_fetch_file(url, "analogy/EN-MSR", verbose=0), "r") as f:
        L = f.read().splitlines()

    # Typical 4 words analogy questions
    questions = []
    answers = []
    category = []
    for l in L:
        words = standardize_string(l).split()
        questions.append(words[0:3])
        answers.append(words[4])
        category.append(words[3])

    verb = set([c for c in set(category) if c.startswith("VB")])
    noun = set([c for c in set(category) if c.startswith("NN")])
    category_high_level = []
    for cat in category:
         if cat in verb:
             category_high_level.append("verb")
         elif cat in noun:
             category_high_level.append("noun")
         else:
             category_high_level.append("adjective")

    assert set([c.upper() for c in category]) == set(['VBD_VBZ', 'VB_VBD', 'VBZ_VBD',
                                         'VBZ_VB', 'NNPOS_NN', 'JJR_JJS', 'JJS_JJR', 'NNS_NN', 'JJR_JJ',
                                         'NN_NNS', 'VB_VBZ', 'VBD_VB', 'JJS_JJ', 'NN_NNPOS', 'JJ_JJS', 'JJ_JJR'])

    return Bunch(X=np.vstack(questions).astype("object"),
                 y=np.hstack(answers).astype("object"),
                 category=np.hstack(category).astype("object"),
                 category_high_level=np.hstack(category_high_level).astype("object"))


# TODO: rewrite to a more standarized version
def fetch_semeval_2012_2(which="all", which_scoring="golden"):
    """
    Fetch dataset used for SEMEVAL 2012 task 2 competition

    Parameters
    -------
    which : "all", "train" or "test"
    which_scoring: "golden" or "platinium" (see Notes)

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X_prot': dictionary keyed on category. Each entry is a matrix of prototype word pairs (see Notes)
        'X': dictionary keyed on category. Each entry is a matrix of question word pairs
        'y': dictionary keyed on category. Each entry is a dictionary word pair -> score

        'categories_names': dictionary keyed on category. Each entry is a human readable name of
        category.
        'categories_descriptions': dictionary keyed on category. Each entry is a human readable description of
        category.

    References
    ----------
    DA Jurgens et al.,
    "Measuring degrees of relational similarity. In *SEM 2012: The First Joint Conference on Lexical
    and Computational Semantics", 2012

    Notes
    -----
    Dataset used in competition was scored as in golden scoring (which_scoring) parameter, however
    organiser have release improved labels afterwards (platinium scoring)

    The task is, given two pairs of words, A:B and C:D, determine the degree to which the semantic relations between
    A and B are similar to those between C and D. Unlike the more familiar task of semantic relation identification,
    which assigns each word pair to a discrete semantic relation class, this task recognizes the continuous range of
    degrees of relational similarity. The challenge is to determine the degrees of relational similarity between a
    given reference word pair and a variety of other pairs, mostly in the same general semantic relation class as the
    reference pair.
    """
    assert which in ['all', 'train', 'test']
    assert which_scoring in ['golden', 'platinium']

    path = _fetch_file(url="https://www.dropbox.com/sh/aarqsfnumx3d8ds/AAB05Mu2HdypP0pudGrNjooaa?dl=1",
                       data_dir="analogy",
                       uncompress=True,
                       move="EN-SEMVAL-2012-2/EN-SEMVAL-2012-2.zip",
                       verbose=0)

    train_files = set(glob.glob(os.path.join(path, "train*.txt"))) - \
                  set(glob.glob(os.path.join(path, "train*_meta.txt")))
    test_files = set(glob.glob(os.path.join(path, "test*.txt"))) - \
                 set(glob.glob(os.path.join(path, "test*_meta.txt")))

    if which == "train":
        files = train_files
    elif which == "test":
        files = test_files
    elif which == "all":
        files = train_files.union(test_files)

    # Every question is formed as similarity to analogy category that is
    # posed as a list of 3 prototype word pairs
    questions = defaultdict(list)
    prototypes = {}
    golden_scores = {}
    platinium_scores = {}
    scores = {"platinium": platinium_scores, "golden": golden_scores}
    categories_names = {}
    categories_descriptions = {}
    for f in files:
        with open(f[0:-4] + "_meta.txt") as meta_f:
            meta = meta_f.read().splitlines()[1].split(",")

        with open(os.path.dirname(f) + "/pl-" + os.path.basename(f)) as f_pl:
            platinium = f_pl.read().splitlines()

        with open(f) as f_gl:
            golden = f_gl.read().splitlines()

        assert platinium[0] == golden[0], ("Incorrect file for ", f)

        c = meta[0] + "_" + meta[1]
        categories_names[c] = meta[2] + "_" + meta[3]
        categories_descriptions[c] = meta[4]

        prototypes[c] = [l.split(":") for l in \
                         platinium[0].replace(": ", ":").replace(" ", ",").replace(".", "").split(",")]
        golden_scores[c] = {}
        platinium_scores[c] = {}
        questions_raw = []
        for line_pl in platinium[1:]:
            word_pair, score = line_pl.split()
            questions_raw.append(word_pair)
            questions[c].append([standardize_string(w) for w in word_pair.split(":")])
            platinium_scores[c][word_pair] = score

        for line_g in golden[1:]:
            word_pair, score = line_g.split()
            golden_scores[c][word_pair] = score

        # Make scores a list
        platinium_scores[c] = [platinium_scores[c][w] for w in questions_raw]
        golden_scores[c] = [golden_scores[c][w] for w in questions_raw]

    return Bunch(X_prot=_change_list_to_np(prototypes),
                 X=_change_list_to_np(questions),
                 y=scores[which_scoring],
                 categories_names=categories_names,
                 categories_descriptions=categories_descriptions)


