Word Embeddings Benchmarks
=====

.. image:: https://travis-ci.com/kudkudak/word-embeddings-benchmarks.svg?token=tTz7fwsU6YC2L4acExst&branch=dev
    :target: https://travis-ci.org/kudkudak/word-embeddings-benchmarks

Word Embedding Benchmark (web) package is focused on very quick assesment of embedding.
The research goal of the package is to quantify what knowledge can be stored in the embeddings.

Goals:

* intuitive scikit-learn like API
* completness (analogy, similarity, categorization and more tasks)
* easy to extend by new tasks and embeddings

Dependencies
======

Please see the requirements.txt and pip_requirements.txt file.

Install
======

This package uses setuptools. You can install it running:

    python setup.py install

If you hve problems during this installation. First you may need to install the dependencies:

    pip install -r requirements.txt

If you already have the dependencies listed in requirements.txt installed,
to install in your home directory, use::

    python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

You can also install it in development mode with::

    python setup.py develop

Examples:
=========

