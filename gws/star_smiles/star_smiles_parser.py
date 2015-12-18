# coding=utf-8

import warnings
from collections import namedtuple

from tokenizer import StarSmilesTokenizer
from tokens import StarSmilesTokens
from parser_data import BondMark, PositionFlag, symbol_to_bond_mark, is_bond_symbol
from parser_exceptions import StarSmilesFormatError

# синтаксис star-smiles
# SSm = end | B | SSB | B SSm |  SSm          # star-smiles
# B   = '{' IBS '}' | '{' IBS '}' SM          # block
# IBS = T | T BM | T IBS | T BM IBS           # in block smiles
# SSB = T | T BM | T SM | T BM SM | T SM BM   # starred smiles block
# BM  = '<' text '>' | '<' '>'                # bond mark
# SM  = '*' | '**' | '***' | ^                # star marks token
# T   = text                                  # text token


StarSmilesBlock = namedtuple('StarSmilesBlock', ['data', 'position_flag'])


class StarSmilesPart(object):
    def __init__(self, smiles, bond_marks=None, position_flag=PositionFlag.none):
        self.smiles = smiles
        self.bond_marks = bond_marks or []
        self.position_flag = position_flag

    def __repr__(self):
        return 'SmilesPart({}, {}, {})'.format(self.smiles, self.bond_marks, self.position_flag)


class StarSmilesParser(object):
    def __init__(self, star_smiles):
        self.tokens = StarSmilesTokenizer(star_smiles)
        self.smiles = None

        self.block_smiles_data = []
        self.parsed_data = []

        self.process_tokens()
        (self.smiles, self.insert_positions,
         self.attach_positions, self.attach_bonds, self.fragment_pos) = self.process_parsed_data()

    @staticmethod
    def process_ss_part(ss_part, insert_pos, attach_pos, attach_bonds, fragment_pos, smiles_parts,
                        smiles_length, position_flag=None, all_atoms_flagged=False):
        smiles_parts.append(ss_part.smiles)
        smiles_length += len(ss_part.smiles)
        if ss_part.bond_marks:
            attach_bonds.extend((smiles_length - 1, bond_mark.value)
                                for bond_mark in ss_part.bond_marks)
        position_flag = position_flag or ss_part.position_flag
        if not all_atoms_flagged:
            positions = [smiles_length - 1]
        else:
            positions = xrange(smiles_length - len(ss_part.smiles), smiles_length)
        if position_flag in (PositionFlag.both, PositionFlag.attach):
            attach_pos.extend(positions)
        if position_flag in (PositionFlag.both, PositionFlag.insert):
            insert_pos.extend(positions)
        if position_flag == PositionFlag.fragment:
            fragment_pos.extend(positions)

        return smiles_length

    def process_parsed_data(self):
        smiles_parts = []
        smiles_length = 0
        insert_pos = []
        attach_pos = []
        fragment_pos = []
        attach_bonds = []
        for data in self.parsed_data:
            if type(data) is StarSmilesPart:
                smiles_length = self.process_ss_part(
                    data, insert_pos, attach_pos, attach_bonds, fragment_pos,
                    smiles_parts, smiles_length)
            elif type(data) is StarSmilesBlock:
                pos_flag = data.position_flag
                for ss_part in data.data:
                    smiles_length = self.process_ss_part(
                        ss_part, insert_pos, attach_pos, attach_bonds, fragment_pos,
                        smiles_parts, smiles_length,
                        position_flag=pos_flag, all_atoms_flagged=True)
        return ''.join(smiles_parts), insert_pos, attach_pos, attach_bonds, fragment_pos

    def process_tokens(self):
        self.parse_star_smiles()

    def parse_star_smiles(self):
        token = self.tokens.peek()
        if token.type == StarSmilesTokens.end_token:
            return
        if token.type == StarSmilesTokens.open_bracket:
            self.parse_block()
        else:
            self.parse_starred_smiles_block()
        self.parse_star_smiles()

    def parse_block(self):
        self.block_smiles_data = []
        self.assert_token(StarSmilesTokens.open_bracket, 'Ожидается {')
        self.parse_in_block_smiles()
        self.assert_token(StarSmilesTokens.close_bracket, 'Ожидается }')
        token = self.tokens.peek()
        if token.type == StarSmilesTokens.stars:
            next(self.tokens)  # skip stars
            stars_count = len(token.data)
            if stars_count == 3:
                warnings.warn(
                    'На позиции {}: три * на блоке можно не указывать. '
                    'Это поведение по умолчанию'.format(token.position))
            positions_flag = self._stars_to_position_flag(token)
        elif token.type == StarSmilesTokens.roof:
            next(self.tokens)  # skip stars
            roof_count = len(token.data)
            if roof_count > 1:
                raise StarSmilesFormatError(
                    StarSmilesTokens.stars, StarSmilesTokens.stars, token.position,
                    'Не допускается больше одного ^ у блока')
            positions_flag = self._roofs_to_position_flag(token)
        else:
            positions_flag = PositionFlag.both
        self.parsed_data.append(StarSmilesBlock(self.block_smiles_data, positions_flag))

    def parse_in_block_smiles(self):
        """
        Разбирает базовое выражение внутри блока {}
        Сохраняет найденные части в block_smiles_data
        """
        token = next(self.tokens)

        if token.type != StarSmilesTokens.text:
            raise StarSmilesFormatError(
                StarSmilesTokens.text, token.type, token.position, 'Ожидается SMILES')
        if self.tokens.peek().type == StarSmilesTokens.close_bracket:
            self.block_smiles_data.append(StarSmilesPart(token.data))
            return
        if self.tokens.peek().type == StarSmilesTokens.open_bond_bracket:
            bond_types = self.parse_bond_marks()
            self.block_smiles_data.append(StarSmilesPart(token.data, bond_marks=bond_types))
        if self.tokens.peek().type == StarSmilesTokens.close_bracket:
            return
        self.parse_in_block_smiles()

    def parse_bond_marks(self):
        """
        Разбирает последовательность <->, <=>, <#>

        :return: тип связи, символ -, = или #
        """
        self.assert_token(StarSmilesTokens.open_bond_bracket, 'Ожидается <')
        if self.tokens.peek().type == StarSmilesTokens.close_bond_bracket:
            warnings.warn('На позиции {}: не найдено обозначение типа связи. '
                          'По умолчанию будет использована одинарная связь "-"'.
                          format(self.tokens.peek().position))
            next(self.tokens)
            return BondMark.single

        text_token = self.assert_token(StarSmilesTokens.text, 'Ожидается обозначение связи аттача')
        self.assert_token(StarSmilesTokens.close_bond_bracket, 'Ожидается >')

        bonds = []
        for i, bond_str in enumerate(text_token.data):
            if not is_bond_symbol(bond_str):
                raise StarSmilesFormatError(
                    StarSmilesTokens.text, StarSmilesTokens.text, text_token.position + i,
                    'Найден неизвестный символ в обозначениях связи: ' + bond_str)
            bonds.append(symbol_to_bond_mark(bond_str))
        return bonds

    def parse_starred_smiles_block(self):
        text_token = next(self.tokens)
        if text_token.type != StarSmilesTokens.text:
            raise StarSmilesFormatError(
                StarSmilesTokens.text, text_token.type, text_token.position, 'Ожидается SMILES')
        if self.tokens.peek().type == StarSmilesTokens.open_bond_bracket:
            bond_type = self.parse_bond_marks()
            if self.tokens.peek().type == StarSmilesTokens.stars:
                position_flag = self._stars_to_position_flag(next(self.tokens))
                self.parsed_data.append(
                    StarSmilesPart(text_token.data,
                                   bond_marks=bond_type,
                                   position_flag=position_flag))
                return
            self.parsed_data.append(StarSmilesPart(text_token.data, bond_marks=bond_type))
            return
        if self.tokens.peek().type == StarSmilesTokens.stars:
            stars_token = next(self.tokens)  # skip stars
            position_flag = self._stars_to_position_flag(stars_token)
            if self.tokens.peek().type == StarSmilesTokens.open_bond_bracket:
                bond_type = self.parse_bond_marks()
                self.parsed_data.append(
                    StarSmilesPart(text_token.data,
                                   bond_marks=bond_type,
                                   position_flag=position_flag))
                return
            self.parsed_data.append(StarSmilesPart(text_token.data, position_flag=position_flag))
            return
        if self.tokens.peek().type == StarSmilesTokens.roof:
            roof_token = next(self.tokens)
            position_flag = self._roofs_to_position_flag(roof_token)
            if self.tokens.peek().type == StarSmilesTokens.open_bond_bracket:
                bond_type = self.parse_bond_marks()
                self.parsed_data.append(
                    StarSmilesPart(text_token.data,
                                   bond_marks=bond_type,
                                   position_flag=position_flag))
                return
            self.parsed_data.append(StarSmilesPart(text_token.data, position_flag=position_flag))
            return
        self.parsed_data.append(StarSmilesPart(text_token.data))

    def assert_token(self, token_type, message):
        """
        :param token_type: тип следующего ожидаемого токена
        :param message: сообщение, если попался другой токен
        :return: ожидаемый токен
        """
        next_token = next(self.tokens)
        if next_token.type != token_type:
            raise StarSmilesFormatError(token_type, next_token.type, next_token.position, message)
        return next_token

    @staticmethod
    def _stars_to_position_flag(stars_token):
        assert len(stars_token.data) == stars_token.data.count('*')
        stars_count = len(stars_token.data)
        if stars_count > 3:
            raise StarSmilesFormatError(
                StarSmilesTokens.stars, StarSmilesTokens.stars, stars_token.position,
                'Не допускается больше трёх * у блока')
        if stars_count == 3:
            return PositionFlag.both
        if stars_count == 2:
            return PositionFlag.insert
        return PositionFlag.attach

    @staticmethod
    def _roofs_to_position_flag(roof_token):
        assert len(roof_token.data) == roof_token.data.count('^')
        roof_count = len(roof_token.data)
        if roof_count > 1:
            raise StarSmilesFormatError(
                StarSmilesTokens.stars, StarSmilesTokens.stars, roof_token.position,
                'Не допускается больше одного ^ у блока')
        return PositionFlag.fragment
