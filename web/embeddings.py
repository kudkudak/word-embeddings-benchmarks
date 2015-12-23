"""
 Fetchers for publicly available pretrained embeddings
"""
from six.moves import cPickle as pickle
from os import path
from .datasets.utils import _get_dataset_dir, _fetch_file
from .embedding import Embedding

def load_embedding(fname, format="word2vec_bin", normalize=True, standardization="lower", load_kwargs={}):
    """
    Loads embeddings from file

    Parameters
    ----------
    fname: string
      Path to file containing embedding

    format: string
      Format of the embedding. Possible values are:
      'word2vec_bin', 'word2vec', 'glove', 'dict'

    normalize: bool, default: True
      If true will normalize all vector to unit length

    standardization: string, default: "lower"
      Standardization to words that will be applied.
      Possible values are: 'normal' and 'lower'. If 'normal' is passed
      words will be stripped of non-alphanumeric characters and if 'lower' is passed
      additionally every word will be lowered.

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.
    """
    assert format in ['word2vec_bin', 'word2vec', 'glove', 'dict'], "Unrecognized format"
    assert standardization in ["lower", "normal", "none", None], "Unrecognized standardization parameter"
    if format == "word2vec_bin":
        w = Embedding.from_word2vec(fname, binary=True)
    elif format == "word2vec":
        w = Embedding.from_word2vec(fname, binary=False)
    elif format == "glove":
        w = Embedding.from_glove(fname, **load_kwargs)
    elif format == "dict":
        d = pickle.load(open(fname, "rb"))
        w = Embedding.from_dict(d)
    if normalize:
        w.normalize_words(inplace=True)
    if standardization=="lower" or standardization=="normal":
        w.standardize_words(lower=(standardization=="lower"), inplace=True)
    return w


def fetch_GloVe(dim=300, corpus="wiki-6B", normalize=True, standardization="lower"):
    """
    Fetches GloVe embeddings. Warning: loading GloVe format is very time consuming

    Parameters
    ----------
    dim: int, default: 300
      Dimensionality of embedding (usually performance increases with dimensionality).
      Available dimensionalities:
        * wiki-6B: 50, 100, 200, 300
        * common-crawl-42B: 300
        * common-crawl-840B: 300
        * twitter: 25, 50, 100, 200

    corpus: string, default: "wiki-6B"
      Corpus that GloVe vector were trained on.
      Available corpuses: "wiki-6B", "common-crawl-42B", "common-crawl-840B", "twitter-27B"

    References
    ----------
    Project website: http://nlp.stanford.edu/projects/glove/
    """
    download_file = {
            "wiki-6B": "http://nlp.stanford.edu/data/glove.6B.zip",
            "common-crawl-42B": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
            "common-crawl-840B": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
            "twitter-27B": "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
    }

    embedding_file = {
        "wiki-6B": {
            50: "glove.6B/glove.6B.50d.txt",
            100: "glove.6B/glove.6B.100d.txt",
            200: "glove.6B/glove.6B.200d.txt",
            300: "glove.6B/glove.6B.300d.txt"
        },
        "common-crawl-42B": {
            300: "glove.42B.300d/glove.42B.300d.txt"
        },
        "common-crawl-840B": {
            300: "glove.840B.300d/glove.840B.300d.txt"
        },
        "twitter-27B": {
            25: "glove.twitter.27B/glove.twitter.27B.25d.txt",
            50: "glove.twitter.27B/glove.twitter.27B.50d.txt",
            100: "glove.twitter.27B/glove.twitter.27B.100d.txt",
            200: "glove.twitter.27B/glove.twitter.27B.200d.txt",
        }
    }

    vocab_size = {
            "wiki-6B": 400000,
            "common-crawl-42B": 1917494,
            "common-crawl-840B": 2196017,
            "twitter-27B": 1193514
    }

    assert corpus in download_file, "Unrecognized corpus"
    assert dim in embedding_file[corpus], "Not available dimensionality"

    _ = _fetch_file(url=download_file[corpus],
                           data_dir="embeddings",
                           uncompress=True,
                           verbose=0)

    return load_embedding(path.join(_get_dataset_dir("embeddings"), embedding_file[corpus][dim]),
                           format="glove",
                           normalize=normalize,
                           standardization=standardization,\
                           load_kwargs={"vocab_size": vocab_size[corpus], "dim": dim})


def fetch_NTM(which="DE", normalize=True, standardization="lower"):
    """
    Fetches word embeddings induced by Neural Translation Machine published by F.Hill
    on https://www.cl.cam.ac.uk/~fh295/

    Parameters
    ----------
    which: str, default: "DE"
      Can choose between DE and FR, which fetches accordingly EN -> DE or EN -> FR translation
      induced word embeddings

    Returns
    -------
    w: Embedding
      Instance of Embedding class
    """
    dirname = _fetch_file(url="https://www.cl.cam.ac.uk/~fh295/TEmbz.tar.gz",
                       data_dir="embeddings",
                       uncompress=True,
                       verbose=0)

    assert which in ["DE", "FR"], "Unrecognized which parameter"

    fname = {"FR": "Trans_embds/D_RNN_500k_144h.pkl", "DE": "Trans_embds/D_german_50k_500k_168h.pkl"}

    return load_embedding(path.join(dirname, fname[which]),
                           format="dict",
                           normalize=normalize,
                           standardization=standardization)



def fetch_PDC(dim=300, normalize=True, standardization="lower"):
    """
    Fetches HDC embeddings trained on wiki by Fei Sun

    Returns
    -------
    dim: int, default:300
      Dimensionality of embedding

    References
    ----------
    Embeddings were published on http://ofey.me/projects/wordrep/.
    Reference paper: Fei Sun, Jiafeng Guo, Yanyan Lan, Jun Xu, and Xueqi Cheng.
    "Learning word representations by jointly modeling syntagmatic and paradigmatic relations"
    """

    url = {
        50: "https://www.dropbox.com/s/0ofi1glri8l42y1/wikicorp.201004-pdc-"
             "iter-20-alpha-0.05-window-10-dim-50-neg-10-subsample-0.0001.txt.bz2?dl=1",
        100: "https://www.dropbox.com/s/fmvegh4j62hulr0/wikicorp.201004-pdc-"
             "iter-20-alpha-0.05-window-10-dim-100-neg-10-subsample-0.0001.txt.bz2?dl=1",
        300: "https://www.dropbox.com/s/jppkd6j2xxb9v48/wikicorp.201004-pdc-"
             "iter-20-alpha-0.05-window-10-dim-300-neg-10-subsample-0.0001.txt.bz2?dl=1"
    }
    assert dim in url, "Unavailable dimensionality"

    path = _fetch_file(url=url[dim],
                        data_dir="embeddings",
                           uncompress=False,
                           move="pdc/pdc{}.txt.bz2".format(dim),
                           verbose=0)

    return load_embedding(path, format="word2vec", normalize=normalize, standardization=standardization)



def fetch_HDC(dim=300, normalize=True, standardization="lower"):
    """
    Fetches PDC embeddings trained on wiki by Fei Sun

    Returns
    -------
    dim: int, default:300
      Dimensionality of embedding

    References
    ----------
    Embeddings were published on http://ofey.me/projects/wordrep/.
    Reference paper: Fei Sun, Jiafeng Guo, Yanyan Lan, Jun Xu, and Xueqi Cheng.
    "Learning word representations by jointly modeling syntagmatic and paradigmatic relations"
    """

    url = {
        50: "https://www.dropbox.com/s/q22ssy8055loknz/wikicorp.201004-hdc-"
            "iter-20-alpha-0.025-window-10-dim-50-neg-10-subsample-0.0001.txt.bz2?dl=1",
        100: "https://www.dropbox.com/s/13226et55fi6g50/wikicorp.201004-hdc-"
             "iter-20-alpha-0.025-window-10-dim-100-neg-10-subsample-0.0001.txt.bz2?dl=1",
        300: "https://www.dropbox.com/s/jrfwel32yd8w0lu/wikicorp.201004-hdc-"
             "iter-20-alpha-0.025-window-10-dim-300-neg-10-subsample-0.0001.txt.bz2?dl=1"
    }
    assert dim in url, "Unavailable dimensionality"

    path = _fetch_file(url=url[dim],
                        data_dir="embeddings",
                           uncompress=False,
                           move="hdc/hdc{}.txt.bz2".format(dim),
                           verbose=0)

    return load_embedding(path, format="word2vec", normalize=normalize, standardization=standardization)



def fetch_SG_GoogleNews(normalize=True, standardization="lower"):
    """
    Fetches SG (skip-gram with negative sampling)
    embeddings trained on GoogleNews dataset published on word2vec website

    Returns
    -------
    w: Embedding
      Instance of Embedding class

    References
    ----------
    Original source: https://code.google.com/p/word2vec/
    """
    path = _fetch_file(url="https://www.dropbox.com/s/bnm0trligffakd9/GoogleNews-vectors-negative300.bin.gz?dl=1",
                           data_dir="embeddings",
                           verbose=0)
    return load_embedding(path, format="word2vec_bin", normalize=normalize, standardization=standardization)


# TODO: uncomment after training is finished
# def fetch_SG_wiki(normalize=True, standardization="lower"):
#     """
#     Fetches SG (skip-gram) embeddings trained on recent (12.2015) Wiki corpus using gensim
#
#     Note
#     ----
#     Doesn't distinguish between lower and capital letters in embedding.
#     See scripts used for training on github in scripts/wikipedia/
#     """
#     fname = path.join(_get_dataset_dir('embeddings'), "sg-wiki-en-400.bin")
#     return _load_embedding(fname, format="word2vec_binary", normalize=normalize,
#                            standardization=standardization)
