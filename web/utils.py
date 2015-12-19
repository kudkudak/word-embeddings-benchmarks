#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for package"""

from os import path
import tarfile
import io
from itertools import islice, chain
from six import string_types, text_type


def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, text_type):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return text_type(text, encoding, errors=errors).encode('utf8')


to_utf8 = any2utf8

# Works just as good with unicode chars
_delchars = [chr(c) for c in range(256)]
_delchars = [x for x in _delchars if not x.isalnum()]
_delchars.remove('\t')
_delchars.remove(' ')
_delchars.remove('_')  # for instance phrases joining in word2vec
_delchars = ''.join(_delchars)
_delchars_table = dict((ord(char), None) for char in _delchars)



# TODO: add support for french and german
def standardize_string(s, remove_nonstandards_chars=True, lower=True, language="english"):
    """
    Ensures common convention across code. Converts to utf-8 and removes non-alphanumeric characters

    Parameters
    -----------
    language: only "english" is now supported. If "english" will remove non-alphanumeric characters
    lower: if True will lower strńing.
    remove_nonstandards_chars: if True will remove non standard characters (for instance '$' or '#')

    Returns
    ----------
    string: processed string
    """

    assert isinstance(s, string_types)

    if not isinstance(s, text_type):
        s = text_type(s, "utf-8")

    if language == "english":
        s = (s.lower() if lower else s)
        s = (s.translate(_delchars_table) if remove_nonstandards_chars else s)
        return s
    else:
        raise NotImplementedError("Not implemented standarization for other languages")


def batched(iterable, size):
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        yield chain([batchiter.next()], batchiter)


def _open(file_, mode='r'):
    """Open file object given filenames, open files or even archives."""
    if isinstance(file_, string_types):
        _, ext = path.splitext(file_)
        if ext in {'.bz2', '.gz'}:
            s = tarfile.open(file_)
            return s.extractfile(s.next())
        else:
            return io.open(file_, mode)
    return file_
