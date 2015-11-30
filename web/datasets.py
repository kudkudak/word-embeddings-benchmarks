"""
 Functions for fetching data
"""

from sklearn.datasets.base import Bunch
from .utils import _get_dataset_dir, _fetch_files
from collections import defaultdict

def fetch_msr_analogy():
    """
    Fetch MSR dataset for testing performance on syntactic analogies

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'question': dictionary keyed on category with lists of space
        delimited question words
        'answers': dictionary keyed on category with the answer

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: Add notes

    """

    dataset_name = 'analogy/EN-MSR'
    data_dir = _get_dataset_dir(dataset_name, data_dir=None,
                                verbose=0)
    url = "https://www.dropbox.com/s/ne0fib302jqbatw/EN-MSR.txt?dl=1"
    raw_data = _fetch_files(data_dir, [("EN-MSR.txt", url, {})],
                            resume=True,
                            verbose=0)[0]

    with open(raw_data, "r") as f:
        L = f.read().splitlines()

    questions = defaultdict(list)
    answers = defaultdict(list)
    for l in L:
        words = l.split()
        questions[words[3]].append(" ".join(words[0:3]))
        answers[words[3]] = words[4]

    assert questions.keys() == answers.keys()
    assert set(questions.keys()) == set(['VBD_VBZ', 'VB_VBD', 'VBZ_VBD',
        'VBZ_VB', 'NNPOS_NN', 'JJR_JJS', 'JJS_JJR', 'NNS_NN', 'JJR_JJ',
        'NN_NNS', 'VB_VBZ', 'VBD_VB', 'JJS_JJ', 'NN_NNPOS', 'JJ_JJS', 'JJ_JJR'])

    return Bunch(questions=questions, answers=answers)

def fetch_google_analogy():
    """
    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'question': dictionary keyed on category with lists of space
        delimited question words
        'answers': dictionary keyed on category with the answer
        'categories_high_level': dictionary keyed on higher level category that
        provides coarse grained grouping of categories

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: Add notes

    """

    dataset_name = 'analogy/EN-GOOGLE'
    data_dir = _get_dataset_dir(dataset_name, data_dir=None,
                                verbose=0)
    url = "https://www.dropbox.com/s/eujtyfb5zem1mim/EN-GOOGLE.txt?dl=1"
    raw_data = _fetch_files(data_dir, [("EN-GOOGLE.txt", url, {})],
                            resume=True,
                            verbose=0)[0]

    with open(raw_data, "r") as f:
        L = f.read().splitlines()

    questions = defaultdict(list)
    answers = defaultdict(list)
    category = None
    for l in L:
        if l.startswith(":"):
            category = l.split()[1]
        else:
            words = l.split()
            questions[category].append(" ".join(words[0:3]))
            answers[category] = words[3]

    assert questions.keys() == answers.keys()
    assert set(questions.keys()) == set(['gram3-comparative', 'gram8-plural', 'capital-common-countries',
     'city-in-state','family', 'gram9-plural-verbs', 'gram2-opposite',
     'currency', 'gram4-superlative', 'gram6-nationality-adjective', 'gram7-past-tense',
     'gram5-present-participle', 'capital-world', 'gram1-adjective-to-adverb'])

    categories_high_level = {
        "syntactic": [c for c in questions if c.startswith("gram")],
        "semantic": [c for c in questions if not c.startswith("gram")]
    }

    return Bunch(questions=questions, answers=answers, categories_high_level=categories_high_level)