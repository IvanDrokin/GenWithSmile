# coding=utf-8

from collections import namedtuple
from tokens import StarSmilesTokens, match_char, is_one_char_token


TokenInfo = namedtuple('TokenInfo', ['type', 'position', 'data'])


class StarSmilesTokenizer(object):
    def __init__(self, star_smiles):
        self.star_smiles = star_smiles
        self.tokens = self._get_tokens()
        self.buffered_token = None

    def __iter__(self):
        return self

    def peek(self):
        if self.buffered_token is None:
            self.buffered_token = next(self.tokens)
        return self.buffered_token

    def next(self):
        result = self.buffered_token if self.buffered_token else next(self.tokens)
        self.buffered_token = None
        return result

    def _get_tokens(self):
        start = None
        token_type = None
        for pos, char in enumerate(self.star_smiles):
            new_token_type = match_char(char)
            if is_one_char_token(char):
                if start is not None:          # считывался длинный токен
                    yield TokenInfo(token_type, start, self.star_smiles[start:pos])
                    start = token_type = None
                yield TokenInfo(new_token_type, pos, data=None)
                continue

            if token_type is None:             # длинный токен до этого не считывался
                start = pos
                token_type = new_token_type
                continue

            if new_token_type != token_type:   # начал считываться другой длинный токен
                yield TokenInfo(token_type, start, self.star_smiles[start:pos])
                token_type = new_token_type
                start = pos
                continue

        if start is not None:   # остался незаписанный длинный токен
            yield TokenInfo(token_type, start, self.star_smiles[start:])
        yield TokenInfo(StarSmilesTokens.end_token, len(self.star_smiles), None)
