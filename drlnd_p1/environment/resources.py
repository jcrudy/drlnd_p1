import os
from fnmatch import fnmatchcase
from toolz import curry, flip

def match_filename(directory, pattern):
    contents = list(filter(curry(flip(fnmatchcase))(pattern), os.listdir(directory)))
    if len(contents) > 1:
        raise ValueError('File pattern is ambiguous.')
    if not contents:
        raise ValueError('File pattern does not match anything.')
    return contents[0]

resources = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
banana = os.path.join(resources, match_filename(resources, 'Banana.*'))