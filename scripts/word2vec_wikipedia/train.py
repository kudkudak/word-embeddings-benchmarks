#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 Usage: ./train.py <path to processed wiki> <path to output file>

 Adapted from http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
"""

import logging
import os.path
import sys
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp1 = sys.argv[1:4]

    # NOTE: it doesn't shuffle data between passes, which might degrade performance
    model = Word2Vec(LineSentence(inp),
                     size=300,
                     negative=5,
                     workers=5)

    model.save_word2vec_format(outp1, binary=False)