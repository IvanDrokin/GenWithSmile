# coding=utf-8

import re
import itertools


def preprocess(raw_meta_smiles):
    """
    Принимает на вход необработанный MetaSMILES и раскрывает независимо 
    все группы вида "{...|...|...}" на отдельные блоки "{...}", возвращая 
    последовательность строк.


    :param raw_meta_smiles: string, необработанная строка в формате MetaSMILES
    :return: generator<string>, полученные строки чистого MetaSMILES формата
    """
    matches = re.finditer(r'{.*?\|.*?}', raw_meta_smiles)
    groups, separators = [], []
    prev_match_end = 0
    for match in matches:
        groups.append(match.group().strip('{}').split('|'))
        separator = raw_meta_smiles[prev_match_end:match.start()]
        separators.append(separator)
        prev_match_end = match.end()
    separators.append(raw_meta_smiles[prev_match_end:])

    for values in itertools.product(*groups):
        yield _join_groups(values, separators)


def _join_groups(group, separators):
    assert len(group) == len(separators) - 1
    result = []
    for sep, value in itertools.izip(separators, group):
        result.append(sep)
        result.append('{')
        result.append(value)
        result.append('}')
    result.append(separators[-1])
    return ''.join(result)