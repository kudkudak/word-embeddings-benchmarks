Word Embeddings Benchmarks
=====

.. image:: https://travis-ci.org/kudkudak/word-embeddings-benchmarks.svg
    :target: https://travis-ci.org/kudkudak/word-embeddings-benchmarks

Word Embedding Benchmark (web) package is focused on providing methods for easy evaluating and reporting
results on common benchmarks (analogy, similarity and categorization).

Research goal of the package is to help drive research in word embeddings by easily accessible reproducible
results (as there is a lot of contradictory results in the literature right now).
This should also help to answer question if we should devise new methods for evaluating word embeddings.

To evaluate your embedding (converted to word2vec or python dict pickle)
on all fast-running benchmarks execute `./scripts/eval_on_all.py <path-to-file>`.
See `here <https://github.com/kudkudak/word-embeddings-benchmarks/wiki>`_ results for embeddings available in the package.

Features:

* scikit-learn API and conventions
* 17 popular datasets
* 6 word embeddings
* methods to solve analogy, similarity and categorization tasks


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


Examples
========
See `examples` folder.
