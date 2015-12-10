#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for package"""

import six
from six import text_type as unicode
from six import string_types

from os import path
import tarfile

def _open(file_, mode='r'):
  """Open file object given filenames, open files or even archives."""
  if isinstance(file_, string_types):
    _, ext = path.splitext(file_)
    if ext in {'.bz2', '.gz'}:
      s = tarfile.open(file_)
      return s.extractfile(s.next())
    else:
      return open(file_, mode)
  return file_