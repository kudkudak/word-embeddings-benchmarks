Word Embeddings Benchmarks
=====

.. image:: https://travis-ci.com/kudkudak/word-embeddings-benchmarks.svg?token=tTz7fwsU6YC2L4acExst&branch=dev
    :target: https://travis-ci.org/kudkudak/word-embeddings-benchmarks

Word Embedding Benchmark (web) package is focused on providing a platform for communicating results in the word
embedding field. The research goal is to understand better landscape of published embeddings.

To evaluate your embedding it is enough to convert it to word2vec format (or python dict pickle)
and run `./scripts/eval_on_all.py <path-to-file>`

Features:

* scikit-learn API and conventions
* 17 popular datasets
* 6 word embeddings (in multiple variations)
* methods to solve analogy, similarity and categorization tasks
* easy to run as scripts


Dependencies
======

Please see the requirements.txt and pip_requirements.txt file.

Install
======

This package uses setuptools. You can install it running:

    python setup.py install

If you have problems during this installation. First you may need to install the dependencies:

    pip install -r requirements.txt

If you already have the dependencies listed in requirements.txt installed,
to install in your home directory, use::

    python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

You can also install it in development mode with::

    python setup.py develop


