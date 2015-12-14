"""
Base class for embedding.

NOTE: This file was adapted from the polyglot package
"""

import logging
import numpy as np

from six import PY2
from six import iteritems

from .utils import _open
from .vocabulary import CountedVocabulary, OrderedVocabulary
from six.moves import cPickle as pickle
from functools import partial
from .utils import standardize_string, to_utf8

logger = logging.getLogger(__name__)


class Embedding(object):
    """ Mapping a vocabulary to a d-dimensional points."""

    def __init__(self, vocabulary, vectors):
        self.vocabulary = vocabulary
        self.vectors = np.asarray(vectors)

        if len(self.vocabulary) != self.vectors.shape[0]:
            raise ValueError("Vocabulary has {} items but we have {} "
                             "vectors. It might be due to standardizing procedure removing some words."
                             .format(len(vocabulary), self.vectors.shape[0]))

        if len(self.vocabulary.words) != len(set(self.vocabulary.words)):
            raise ValueError("Vocabulary has duplicates. Most likely standardizing words has introduced duplicates")

    def __getitem__(self, k):
        return self.vectors[self.vocabulary[k]]

    def __contains__(self, k):
        return k in self.vocabulary

    def __delitem__(self, k):
        """Remove the word and its vector from the embedding.

        Note:
         This operation costs \\theta(n). Be careful putting it in a loop.
        """
        index = self.vocabulary[k]
        del self.vocabulary[k]
        self.vectors = np.delete(self.vectors, index, 0)

    def __len__(self):
        return len(self.vocabulary)

    def __iter__(self):
        for w in self.vocabulary:
            yield w, self[w]

    @property
    def words(self):
        return self.vocabulary.words

    @property
    def shape(self):
        return self.vectors.shape

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError as e:
            return default

    def standardize_words(self, lower=False, inplace=False):
        return self.transform_words(partial(standardize_string, lower=lower, remove_nonstandards_chars=True),
                                    inplace=inplace)

    def transform_words(self, f, inplace=False, lower=False):
        """ Tranform words in vocabulary """

        id_map = {}
        word_count = len(self.vectors)
        if inplace:
            for id, w in enumerate(self.vocabulary.words):
                if len(f(w)) and f(w) not in id_map:
                    id_map[f(w)] = id
                    self.vectors[len(id_map) - 1] = self.vectors[id]
            words = sorted(id_map.keys(), key=lambda x: id_map[x])
            self.vectors = self.vectors[0:len(id_map)]
            self.vocabulary = self.vocabulary.__class__(words)
            logger.info("Tranformed {} into {} words".format(word_count, len(words)))
            return self
        else:
            for id, w in enumerate(self.vocabulary.words):
                if len(f(w)) and f(w) not in id_map:
                    id_map[f(w)] = id
            words = sorted(id_map.keys(), key=lambda x: id_map[x])
            vectors = self.vectors[[id_map[w] for w in words]]
            logger.info("Tranformed {} into {} words".format(word_count, len(words)))
            return Embedding(vectors=vectors, vocabulary=self.vocabulary.__class__(words))

    def most_frequent(self, k, inplace=False):
        """Only most frequent k words to be included in the embeddings."""
        vocabulary = self.vocabulary.most_frequent(k)
        vectors = np.asarray([self[w] for w in vocabulary])
        if inplace:
            self.vocabulary = vocabulary
            self.vectors = vectors
            return self
        return Embedding(vectors=vectors, vocabulary=vocabulary)

    def normalize_words(self, ord=2, inplace=False):
        """Normalize embeddings matrix row-wise.

        Parameters:
          ord: normalization order. Possible values {1, 2, 'inf', '-inf'}
        """
        if ord == 2:
            ord = None  # numpy uses this flag to indicate l2.
        vectors = self.vectors.T / np.linalg.norm(self.vectors, ord, axis=1)
        if inplace:
            self.vectors = vectors.T
            return self
        return Embedding(vectors=vectors.T, vocabulary=self.vocabulary)

    def nearest_neighbors(self, word, top_k=10):
        """
        Return the nearest k words to the given `word`.

        Args:
          word (string): single word.
          top_k (integer): decides how many neighbors to report.

        Returns:
          A list of words sorted by the distances. The closest is the first.

        Note:
          L2 metric is used to calculate distances.
        """
        # TODO(rmyeid): Use scikit ball tree, if scikit is available
        point = self[word]
        diff = self.vectors - point
        distances = np.linalg.norm(diff, axis=1)
        top_ids = distances.argsort()[1:top_k + 1]
        return [self.vocabulary.id_word[i] for i in top_ids]

    def distances(self, word, words):
        """Calculate eucledean pairwise distances between `word` and `words`.

        Parameters:
          word (string): single word.
          words (list): list of strings.

        Returns:
          numpy array of the distances.

        Note:
          L2 metric is used to calculate distances.
        """

        point = self[word]
        vectors = np.asarray([self[w] for w in words])
        diff = vectors - point
        distances = np.linalg.norm(diff, axis=1)
        return distances

    @staticmethod
    def from_gensim(model):
        word_count = {}
        vectors = []
        for word, vocab in sorted(iteritems(model.vocab), key=lambda item: -item[1].count):
            word = standardize_string(word)
            if word:
                vectors.append(model.syn0[vocab.index])
                word_count[word] = vocab.count
        vocab = CountedVocabulary(word_count=word_count)
        vectors = np.asarray(vectors)
        return Embedding(vocabulary=vocab, vectors=vectors)

    @staticmethod
    def from_word2vec_vocab(fvocab):
        counts = {}
        with _open(fvocab) as fin:
            for line in fin:
                word, count = standardize_string(line).strip().split()
                if word:
                    counts[word] = int(count)
        return CountedVocabulary(word_count=counts)

    @staticmethod
    def _from_word2vec_binary(fname):
        with _open(fname) as fin:
            # TODO: merge words, words_seen and vectors
            words = []
            header = unicode(fin.readline())
            vocab_size, layer1_size = list(map(int, header.split()))  # throws for invalid file format
            vectors = np.zeros((vocab_size, layer1_size), dtype=np.float32)
            binary_len = np.dtype("float32").itemsize * layer1_size
            index = 0
            for line_no in xrange(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                        word.append(ch)

                words.append(b''.join(word))
                index = line_no
                vectors[index, :] = np.fromstring(fin.read(binary_len), dtype=np.float32)

            return words, vectors

    @staticmethod
    def _from_word2vec_text(fname):
        with _open(fname, 'rb') as fin:
            # TODO: merge words, words_seen and vectors
            words = []
            header = unicode(fin.readline())
            vocab_size, layer1_size = list(map(int, header.split()))  # throws for invalid file format
            vectors = np.zeros(shape=(vocab_size, layer1_size), dtype=np.float32)
            for line_no, line in enumerate(fin):
                try:
                    parts = unicode(line, encoding="utf-8").strip().split()
                except TypeError as e:
                    parts = line.strip().split()
                except Exception as e:
                    import pdb
                    pdb.set_trace()
                    logger.warning("We ignored line number {} because of erros in parsing"
                                   "\n{}".format(line_no, e))
                    continue
                # We differ from Gensim implementation.
                # Our assumption that a difference of one happens because of having a
                # space in the word.
                if len(parts) == layer1_size + 1:
                    word, vectors[line_no] = parts[0], list(map(np.float32, parts[1:]))
                elif len(parts) == layer1_size + 2:
                    word, vectors[line_no] = parts[:2], list(map(np.float32, parts[2:]))
                    word = u" ".join(word)
                else:
                    logger.warning("We ignored line number {} because of unrecognized "
                                   "number of columns {}".format(line_no, parts[:-layer1_size]))
                    continue

                words.append(word)

            return words, vectors

    @staticmethod
    def to_word2vec(w, fname, binary=False):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        w: Embedding instance

        fname: string
          Destination file
        """
        logger.info("storing %sx%s projection weights into %s" % (w.vectors.shape[0], w.vectors.shape[1], fname))
        with _open(fname, 'wb') as fout:
            fout.write(to_utf8("%s %s\n" % w.vectors.shape))
            # store in sorted order: most frequent words at the top
            for word, vector in zip(w.vocabulary.words, w.vectors):
                if binary:
                    fout.write(to_utf8(word) + b" " + vector.tostring())
                else:
                    fout.write(to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in vector))))

    @staticmethod
    def from_word2vec(fname, fvocab=None, binary=False):
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.

        Note that the information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        `binary` is a boolean indicating whether the data is in binary word2vec format.
        Word counts are read from `fvocab` filename, if set (this is the file generated
        by `-save-vocab` flag of the original C tool).
        """
        vocabulary = None
        if fvocab is not None:
            logger.info("loading word counts from %s" % (fvocab))
            vocabulary = Embedding.from_word2vec_vocab(fvocab)

        logger.info("loading projection weights from %s" % (fname))
        if binary:
            words, vectors = Embedding._from_word2vec_binary(fname)
        else:
            words, vectors = Embedding._from_word2vec_text(fname)

        if not vocabulary:
            vocabulary = OrderedVocabulary(words=words)

        return Embedding(vocabulary=vocabulary, vectors=vectors)

    @staticmethod
    def load(fname):
        """Load an embedding dump generated by `save`"""

        content = _open(fname).read()
        if PY2:
            state = pickle.loads(content)
        else:
            state = pickle.loads(content, encoding='latin1')
        voc, vec = state
        if len(voc) == 2:
            words, counts = voc
            word_count = dict(zip(words, counts))
            vocab = CountedVocabulary(word_count=word_count)
        else:
            vocab = OrderedVocabulary(voc)
        return Embedding(vocabulary=vocab, vectors=vec)

    def save(self, fname):
        """Save a pickled version of the embedding into `fname`."""

        vec = self.vectors
        voc = self.vocabulary.getstate()
        state = (voc, vec)
        with open(fname, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
