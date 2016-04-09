# -*- coding: utf-8 -*-
"""
 Fetchers for publicly available pretrained embeddings
"""
from six.moves import cPickle as pickle
from os import path
from .datasets.utils import _get_dataset_dir, _fetch_file
from .embedding import Embedding

def load_embedding(fname, format="word2vec_bin", normalize=True,
                   lower=False, clean_words=False, load_kwargs={}):
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

    clean_words: bool, default: True
      If true will only keep alphanumeric characters and "_", "-"
      Warning: shouldn't be applied to embeddings with non-ascii characters

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.
    """
    assert format in ['word2vec_bin', 'word2vec', 'glove', 'dict'], "Unrecognized format"
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
    if lower or clean_words:
        w.standardize_words(lower=lower, clean_words=clean_words, inplace=True)
    return w



def fetch_GloVe(dim=300, corpus="wiki-6B", normalize=True, lower=False, clean_words=True):
    """
    Fetches GloVe embeddings.

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

    normalize: bool, default: True
      If true will normalize all vector to unit length

    clean_words: bool, default: True
      If true will only keep alphanumeric characters and "_", "-"
      Warning: shouldn't be applied to embeddings with non-ascii characters

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.

    Returns
    -------
    w: Embedding
      Embedding instance

    References
    ----------
    Project website: http://nlp.stanford.edu/projects/glove/

    Notes
    -----
    Loading GloVe format can take a while
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
                           verbose=1)

    return load_embedding(path.join(_get_dataset_dir("embeddings"), embedding_file[corpus][dim]),
                           format="glove",
                           normalize=normalize,
                           lower=lower, clean_words=clean_words,\
                           load_kwargs={"vocab_size": vocab_size[corpus], "dim": dim})



def fetch_HPCA(which, normalize=True, lower=False, clean_words=False):
    """
    Fetches Hellinger PCA based embeddings

    Parameters
    ----------
    which: str, default: "autoencoder_phrase_hpca"
      Can choose between "hpca" and "autoencoder_phrase_hpca" (from "The Sum of Its Parts")

    normalize: bool, default: True
      If true will normalize all vector to unit length

    clean_words: bool, default: True
      If true will only keep alphanumeric characters and "_", "-"
      Warning: shouldn't be applied to embeddings with non-ascii characters

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.

    Returns
    -------
    w: Embedding
      Instance of Embedding class

    References
    ----------
    Published at http://lebret.ch/words/
    Reference paper: Lebret, Collobert et al., “The Sum of Its Parts”: Joint Learning of Word and Phrase Representations with Autoencoders", 2015
    """
    download_file = {
            "autoencoder_phrase_hpca": "https://www.dropbox.com/s/6dyf48crdmjbw1a/AHPCA.bin.gz?dl=1",
            "hpca": "https://www.dropbox.com/s/5y5l6vyn8yn11dv/HPCA.bin.gz?dl=1"
    }

    path = _fetch_file(url=download_file[which],
                        data_dir="embeddings",
                           uncompress=False,
                           verbose=1)

    return load_embedding(path, format="word2vec_bin", normalize=normalize, lower=lower, clean_words=clean_words)



def fetch_morphoRNNLM(which, normalize=True, lower=False, clean_words=False):
    """
    Fetches recursive morphological neural network embeddings

    Parameters
    ----------
    which: str, default: "CW"
      Can choose between CW and HSMN

    normalize: bool, default: True
      If true will normalize all vector to unit length

    clean_words: bool, default: True
      If true will only keep alphanumeric characters and "_", "-"
      Warning: shouldn't be applied to embeddings with non-ascii characters

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.

    Returns
    -------
    w: Embedding
      Instance of Embedding class

    References
    ----------
    Published at http://stanford.edu/~lmthang/morphoNLM/
    Reference paper: Luong, Socher et al., "Better Word Representations with Recursive Neural Networks for Morphology", 2013
    """
    download_file = {
            "CW": "https://www.dropbox.com/s/7fdj2666iqv4xbu/cwCsmRNN.bin.gz?dl=1",
            "HSMN": "https://www.dropbox.com/s/okw1i6kc6e2jd1q/hsmnCsmRNN.bin.gz?dl=1"
    }

    path = _fetch_file(url=download_file[which],
                        data_dir="embeddings",
                           uncompress=False,
                           verbose=1)

    return load_embedding(path, format="word2vec_bin", normalize=normalize, lower=lower, clean_words=clean_words)





def fetch_NMT(which="DE", normalize=True, lower=False, clean_words=False):
    """
    Fetches word embeddings induced by Neural Translation Machine

    Parameters
    ----------
    which: str, default: "DE"
      Can choose between DE and FR, which fetches accordingly EN -> DE or EN -> FR translation
      induced word embeddings

    normalize: bool, default: True
      If true will normalize all vector to unit length

    clean_words: bool, default: True
      If true will only keep alphanumeric characters and "_", "-"
      Warning: shouldn't be applied to embeddings with non-ascii characters

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.

    Returns
    -------
    w: Embedding
      Instance of Embedding class

    References
    ----------
    Published at https://www.cl.cam.ac.uk/~fh295/.
    Reference paper: Hill, Cho et al., "Embedding Word Similarity With Neural Machine Translation", 2014
    """
    dirname = _fetch_file(url="https://www.cl.cam.ac.uk/~fh295/TEmbz.tar.gz",
                       data_dir="embeddings",
                       uncompress=True,
                       verbose=1)

    assert which in ["DE", "FR"], "Unrecognized which parameter"

    fname = {"FR": "Trans_embds/D_RNN_500k_144h.pkl", "DE": "Trans_embds/D_german_50k_500k_168h.pkl"}

    return load_embedding(path.join(dirname, fname[which]),
                           format="dict",
                           normalize=normalize,
                           lower=lower, clean_words=clean_words)



def fetch_PDC(dim=300, normalize=True, lower=False, clean_words=True):
    """
    Fetches PDC embeddings trained on wiki by Fei Sun

    Parameters
    ----------
    dim: int, default:300
      Dimensionality of embedding

    normalize: bool, default: True
      If true will normalize all vector to unit length

    clean_words: bool, default: True
      If true will only keep alphanumeric characters and "_", "-"
      Warning: shouldn't be applied to embeddings with non-ascii characters

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.

    Returns
    -------
    w: Embedding
      Embedding instance

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
                           verbose=1)

    return load_embedding(path, format="word2vec", normalize=normalize, lower=lower, clean_words=clean_words)


def fetch_HDC(dim=300, normalize=True, lower=False, clean_words=True):
    """
    Fetches PDC embeddings trained on wiki by Fei Sun

    Parameters
    ----------
    dim: int, default:300
      Dimensionality of embedding

    normalize: bool, default: True
      If true will normalize all vector to unit length

    clean_words: bool, default: True
      If true will only keep alphanumeric characters and "_", "-"
      Warning: shouldn't be applied to embeddings with non-ascii characters

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.

    Returns
    -------
    w: Embedding
      Embedding instance

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
                           verbose=1)

    return load_embedding(path, format="word2vec", normalize=normalize, lower=lower, clean_words=clean_words)



def fetch_SG_GoogleNews(normalize=True, lower=False, clean_words=True):
    """
    Fetches SG (skip-gram with negative sampling)
    embeddings trained on GoogleNews dataset published on word2vec website

    Parameters
    ----------
    normalize: bool, default: True
      If true will normalize all vector to unit length

    clean_words: bool, default: True
      If true will only keep alphanumeric characters and "_", "-"
      Warning: shouldn't be applied to embeddings with non-ascii characters

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.

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
                           verbose=1)
    return load_embedding(path, format="word2vec_bin", normalize=normalize, lower=lower, clean_words=clean_words)


# TODO: uncomment after training is finished
# def fetch_SG_wiki(normalize=True, lower=False, clean_words=True):
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
#                            lower=lower, clean_words=clean_words)
