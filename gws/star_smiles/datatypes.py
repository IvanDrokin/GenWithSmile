from enum import Enum
from collections import namedtuple


TokenInfo = namedtuple('TokenInfo', ['type', 'position', 'data'])


class PositionFlag(Enum):
    none = 0
    attach = 1
    insert = 2
    both = 3


class BondMark(Enum):
    none = ''
    single = '-'
    double = '='
    triple = '#'


class StarSmilesTokens(Enum):
    text = 0
    open_bracket = 1
    close_bracket = 2
    stars = 3
    open_bond_bracket = 4
    close_bond_bracket = 5
    end_token = 6


class StarSmilesTokensHelper(object):
    _char_tokens = {
        '{': StarSmilesTokens.open_bracket,
        '}': StarSmilesTokens.close_bracket,
        '<': StarSmilesTokens.open_bond_bracket,
        '>': StarSmilesTokens.close_bond_bracket,
    }

    @staticmethod
    def is_one_char_token(char):
        return char in StarSmilesTokensHelper._char_tokens

    @staticmethod
    def match_char(char):
        if char in StarSmilesTokensHelper._char_tokens:
            return StarSmilesTokensHelper._char_tokens[char]
        if '*' in char:
            return StarSmilesTokens.stars
        return StarSmilesTokens.text
