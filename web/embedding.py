"""
Base class for embedding.

NOTE: This file was adapted from the polyglot package
"""

import logging
import numpy as np

from six import text_type
from six import PY2
from six import iteritems
from six import string_types
from .utils import _open
from .vocabulary import Vocabulary, CountedVocabulary, OrderedVocabulary
from six.moves import cPickle as pickle
from six.moves import range
from functools import partial
from .utils import standardize_string, to_utf8

from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)

class Embedding(object):
    """ Mapping a vocabulary to a d-dimensional points."""

    def __init__(self, vocabulary, vectors):
        self.vocabulary = vocabulary
        self.vectors = np.asarray(vectors)
        if len(self.vocabulary) != self.vectors.shape[0]:
            raise ValueError("Vocabulary has {} items but we have {} "
                             "vectors."
                             .format(len(vocabulary), self.vectors.shape[0]))

        if len(self.vocabulary.words) != len(set(self.vocabulary.words)):
            logger.warning("Vocabulary has duplicates.")

    def __getitem__(self, k):
        return self.vectors[self.vocabulary[k]]

    def __setitem__(self, k, v):
        if not v.shape[0] == self.vectors.shape[1]:
            raise RuntimeError("Please pass vector of len {}".format(self.vectors.shape[1]))

        if k not in self.vocabulary:
            self.vocabulary.add(k)
            self.vectors = np.vstack([self.vectors, v.reshape(1,-11)])
        else:
            self.vectors[self.vocabulary[k]] = v


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

    def standardize_words(self, lower=False, clean_words=False, inplace=False):
        return self.transform_words(partial(standardize_string, lower=lower, clean_words=clean_words),
                                    inplace=inplace)

    def transform_words(self, f, inplace=False):
        """ Tranform words in vocabulary """
        id_map = {}
        id_map_to_new = {}
        word_count = len(self.vectors)
        if inplace:
            for id, w in enumerate(self.vocabulary.words):
                fw = f(w)
                if len(fw) and (fw not in id_map):
                    id_map[fw] = id
                    id_map_to_new[fw] = len(id_map) - 1
                    self.vectors[len(id_map) - 1] = self.vectors[id]
                elif len(fw) and (fw in id_map) and (fw == w):
                    # Overwrites with last occurence
                    self.vectors[id_map_to_new[fw]] = self.vectors[id]


            words = sorted(id_map.keys(), key=lambda x: id_map[x])
            self.vectors = self.vectors[0:len(id_map)]
            self.vocabulary = self.vocabulary.__class__(words)
            logger.info("Tranformed {} into {} words".format(word_count, len(words)))
            return self
        else:
            for id, w in enumerate(self.vocabulary.words):
                if len(f(w)) and (f(w) not in id_map or f(w) == w):
                    id_map[f(w)] = id
            words = sorted(id_map.keys(), key=lambda x: id_map[x])
            vectors = self.vectors[[id_map[w] for w in words]]
            logger.info("Tranformed {} into {} words".format(word_count, len(words)))
            return Embedding(vectors=vectors, vocabulary=self.vocabulary.__class__(words))

    def most_frequent(self, k, inplace=False):
        """Only most frequent k words to be included in the embeddings."""

        assert isinstance(self.vocabulary, OrderedVocabulary), \
            "most_frequent can be called only on Embedding with OrderedVocabulary"

        vocabulary = self.vocabulary.most_frequent(k)
        vectors = np.asarray([self[w] for w in vocabulary])
        if inplace:
            self.vocabulary = vocabulary
            self.vectors = vectors
            return self
        return Embedding(vectors=vectors, vocabulary=vocabulary)

    def normalize_words(self, ord=2, inplace=False):
        """Normalize embeddings matrix row-wise.

        Parameters
        ----------
          ord: normalization order. Possible values {1, 2, 'inf', '-inf'}
        """
        if ord == 2:
            ord = None  # numpy uses this flag to indicate l2.
        vectors = self.vectors.T / np.linalg.norm(self.vectors, ord, axis=1)
        if inplace:
            self.vectors = vectors.T
            return self
        return Embedding(vectors=vectors.T, vocabulary=self.vocabulary)


    def nearest_neighbors(self, word, k=1, exclude=[], metric="cosine"):
        """
        Find nearest neighbor of given word

        Parameters
        ----------
          word: string or vector
            Query word or vector.

          k: int, default: 1
            Number of nearest neihbours to return.

          metric: string, default: 'cosine'
            Metric to use.

          exclude: list, default: []
            Words to omit in answer

        Returns
        -------
          n: list
            Nearest neighbors.
        """
        if isinstance(word, string_types):
            assert word in self, "Word not found in the vocabulary"
            v = self[word]
        else:
            v = word

        D = pairwise_distances(self.vectors, v.reshape(1, -1), metric=metric)

        if isinstance(word, string_types):
            D[self.vocabulary.word_id[word]] = D.max()

        for w in exclude:
            D[self.vocabulary.word_id[w]] = D.max()

        return [self.vocabulary.id_word[id] for id in D.argsort(axis=0).flatten()[0:k]]

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
        with _open(fname, 'rb') as fin:
            words = []
            header = fin.readline()
            vocab_size, layer1_size = list(map(int, header.split()))  # throws for invalid file format
            logger.info("Loading #{} words with {} dim".format(vocab_size, layer1_size))
            vectors = np.zeros((vocab_size, layer1_size), dtype=np.float32)
            binary_len = np.dtype("float32").itemsize * layer1_size
            for line_no in range(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                        word.append(ch)

                words.append(b''.join(word).decode("latin-1"))
                vectors[line_no, :] = np.fromstring(fin.read(binary_len), dtype=np.float32)

            if len(words) < vocab_size:
                logger.warning("Omitted {} words".format(vocab_size - len(words)))
            elif len(words) > vocab_size:
                raise RuntimeError("Read too many words, incorrect file")

            return words, vectors

    @staticmethod
    def _from_word2vec_text(fname):
        with _open(fname, 'r') as fin:
            words = []
            header = fin.readline()
            ignored = 0
            vocab_size, layer1_size = list(map(int, header.split()))  # throws for invalid file format
            vectors = np.zeros(shape=(vocab_size, layer1_size), dtype=np.float32)
            for line_no, line in enumerate(fin):
                try:
                    parts = text_type(line, encoding="utf-8").strip().split()
                except TypeError as e:
                    parts = line.strip().split()
                except Exception as e:
                    logger.warning("We ignored line number {} because of erros in parsing"
                                   "\n{}".format(line_no, e))
                    continue
                # We differ from Gensim implementation.
                # Our assumption that a difference of one happens because of having a
                # space in the word.
                if len(parts) == layer1_size + 1:
                    word, vectors[line_no - ignored] = parts[0], list(map(np.float32, parts[1:]))
                elif len(parts) == layer1_size + 2:
                    word, vectors[line_no - ignored] = parts[:2], list(map(np.float32, parts[2:]))
                    word = u" ".join(word)
                else:
                    ignored += 1
                    logger.warning("We ignored line number {} because of unrecognized "
                                   "number of columns {}".format(line_no, parts[:-layer1_size]))
                    continue

                words.append(word)
            if ignored:
                vectors = vectors[0:-ignored]

            if len(words) < vocab_size:
                logger.warning("Omitted {} words".format(vocab_size - len(words)))
            elif len(words) > vocab_size:
                raise RuntimeError("Read too many words, incorrect file")

            return words, vectors


    @staticmethod
    def from_glove(fname, vocab_size, dim):
        with _open(fname, 'r') as fin:
            words = []
            ignored = 0
            vectors = np.zeros(shape=(vocab_size, dim), dtype=np.float32)
            for line_no, line in enumerate(fin):
                try:
                    parts = text_type(line, encoding="utf-8").strip().split()
                except TypeError as e:
                    parts = line.strip().split()
                except Exception as e:
                    ignored += 1
                    logger.warning("We ignored line number {} because of erros in parsing"
                                   "\n{}".format(line_no, e))
                    continue

                try:
                    word, vectors[line_no - ignored] = " ".join(parts[0:len(parts) - dim]), list(map(np.float32, parts[len(parts) - dim:]))
                    words.append(word)
                except Exception as e:
                    ignored += 1
                    logger.warning("We ignored line number {} because of erros in parsing"
                                   "\n{}".format(line_no, e))

            return Embedding(vocabulary=OrderedVocabulary(words), vectors=vectors[0:len(words)])


    @staticmethod
    def from_dict(d):
        for k in d: # Standarize
            d[k] = np.array(d[k]).flatten()
        return Embedding(vectors=list(d.values()), vocabulary=Vocabulary(d.keys()))

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
                    fout.write(to_utf8(word) + b" " + vector.astype("float32").tostring())
                else:
                    fout.write(to_utf8("%s %s\n" % (word, ' '.join("%.15f" % val for val in vector))))

    @staticmethod
    def from_word2vec(fname, fvocab=None, binary=False):
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.

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

        if len(words) != len(set(words)):
            raise RuntimeError("Vocabulary has duplicates")

        e = Embedding(vocabulary=vocabulary, vectors=vectors)

        return e

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
