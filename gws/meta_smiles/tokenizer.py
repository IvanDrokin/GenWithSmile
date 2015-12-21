# coding=utf-8

from collections import namedtuple
from tokens import Token, match_char, is_one_char_token


""" 
Хранит информацию о токене: 
  :param type: meta_smiles.tokens.Token -- тип токена
  :param position: int -- позиция начала токена
  :param data: str -- содержимое токена, если предполагается типом, None иначе
"""
TokenInfo = namedtuple('TokenInfo', ['type', 'position', 'data'])


class Tokenizer(object):
    """
    Позволяет преобразовывать строку в формате MetaSMILES в последовательность 
    токенов для дальнейшей обработки в парсере. 

    Поддерживает буферизованную на один элемент генерацию токенов. 
    Проход по списку токенов возможен только один раз.
    """
    def __init__(self, meta_smiles):
        self.meta_smiles = meta_smiles
        self._tokens = self._get_tokens()
        self._buffered_token = None

    def __iter__(self):
        return self

    def next(self):
        result = self._buffered_token if self._buffered_token else next(self._tokens)
        self._buffered_token = None
        return result

    def peek(self):
        """
        :return: элемент, который вернётся при следующем вызове функции next
        """
        if self._buffered_token is None:
            self._buffered_token = next(self._tokens)
        return self._buffered_token

    def _get_tokens(self):
        """
        Не буферизующий генератор токенов
        """
        start = None
        token_type = None
        for pos, char in enumerate(self.meta_smiles):
            new_token_type = match_char(char)
            if is_one_char_token(char):
                if start is not None:          # считывался длинный токен
                    yield TokenInfo(token_type, start, self.meta_smiles[start:pos])
                    start = token_type = None
                yield TokenInfo(new_token_type, pos, data=None)
                continue

            if token_type is None:             # длинный токен до этого не считывался
                start = pos
                token_type = new_token_type
                continue

            if new_token_type != token_type:   # начал считываться другой длинный токен
                yield TokenInfo(token_type, start, self.meta_smiles[start:pos])
                token_type = new_token_type
                start = pos
                continue

        if start is not None:   # остался незаписанный длинный токен
            yield TokenInfo(token_type, start, self.meta_smiles[start:])
        yield TokenInfo(Token.end, len(self.meta_smiles), None)
