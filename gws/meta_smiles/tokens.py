# coding: utf-8
from enum import Enum


class Token(Enum):
    """
    Класс, описывающий все токены MetaSMILES формата
    """
    text = 0
    left_curly_bracket = 1
    right_curly_bracket = 2
    colon = 3
    vertical_line = 4
    end = 5


_char_tokens = {
    '{': Token.left_curly_bracket,
    '}': Token.right_curly_bracket,
    ':': Token.colon,
    '|': Token.vertical_line
}


def is_one_char_token(char):
    """
    :return: true, если символ char является односимвольным токеном, иначе false
    """
    return char in _char_tokens


def match_char(char):
    """
    :return: токен, соответствующий переданному символу char
    """
    if is_one_char_token(char):
        return _char_tokens[char]
    return Token.text
