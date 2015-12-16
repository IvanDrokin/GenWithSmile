from enum import Enum


class StarSmilesTokens(Enum):
    text = 0
    open_bracket = 1
    close_bracket = 2
    stars = 3
    open_bond_bracket = 4
    close_bond_bracket = 5
    end_token = 6
    roof = 7

_char_tokens = {
    '{': StarSmilesTokens.open_bracket,
    '}': StarSmilesTokens.close_bracket,
    '<': StarSmilesTokens.open_bond_bracket,
    '>': StarSmilesTokens.close_bond_bracket,
}


def is_one_char_token(char):
    return char in _char_tokens


def match_char(char):
    if char in _char_tokens:
        return _char_tokens[char]
    if '*' in char:
        return StarSmilesTokens.stars
    if '^' in char:
        return StarSmilesTokens.roof
    return StarSmilesTokens.text
