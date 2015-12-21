# coding=utf-8

import warnings
from collections import namedtuple

from enum import Enum

from tokenizer import Tokenizer
from tokens import Token
from parser_exceptions import MetaSmilesFormatError

# Синтаксис SMILES: http://opensmiles.org/opensmiles.html

# Примеры корректных MetaSMILES: 
#   C1C(O)-{remove:1}C1
#   CC{attach:fixed:-=#}={remove:1}CN


### Примеры разбора строки MetaSMILES:
## Пример со связью, разделённой веткой с атомом
## C1C{insert:fixed}(O{attach:fixed:-})-{remove:1}C1
# после парсинга: 
#         012345678
# smiles: C1C(O)-C1
# atom_labels: [
#     type: attach, pos: 4, requirement: fixed, bonds: [SINGLE]
#     type: insert, pos: 2, requirement: fixed
# ]
# bond_labels: [
#     type: remove, pos: 6, count: 1
# ]
#
# После разбора SMILES:
# atom_id: 0 1 2 3
# smiles:  C1C(O)C1
# atom_labels: [
#     type: attach, atom: 2, requirement: fixed, bonds: [SINGLE]
#     type: insert, atom: 1, requirement: fixed
# ]
# bond_labels: [
#     type: remove, atoms: (1, 3), count: 1
# ]

## Пример со связью, связывающей разрыв цикла, исходная молекула C=1CCCCC=1
## Различные способы записать пометку двойной связи цикла на разрыв:
## C={remove:2}1CCCCC={remove:2}1
## C={remove:2}1CCCCC=1
## C=1CCCCC={remove:2}1
## C={remove:2}1CCCCC1
## C1CCCCC={remove:2}1

## Неправильные способы задания меток связей:
## C={remove:1}1CCCCC={remove:2}1   # различаются метки

## Пример на основе последовательности C={remove:2}1CCCCC={remove:2}1
# После парсинга:
#         0123456789
# smiles: C=1CCCCC=1
# atom_labels: []
# bond_labels: [
#     type: remove, pos: 1, count: 2
#     type: remove, pos: 8, count: 2
# ]
#
# После разбора SMILES:
# atom_id: 0  12345
# smiles:  C=1CCCCC=1
# atom_labels = []
# bond_labels = [
#     type: remove, atoms: (0, 5), count: 2
# ]

## C1CCF/{remove:1}C=C/FCC1
## NS(O)(={remove:1}O)CN
## CC{attach:fixed:-=}{insert:fixed}CN


"""
Хранит информацию о результате парсинга.
  :param smiles: str -- строка, остающаяся после исключения конструкций формата MetaSMILES
  :param atom_labels: list -- список меток атомов. Типы меток атомов описаны в классах Atom*Label
  :param bond_labels: list -- список меток связей. Типы меток связей описаны в классах Bond*Label
"""
ParseResult = namedtuple('ParseResult', ['smiles', 'atom_labels', 'bond_labels'])


class AtomLabelType(Enum):
    """
    Список допустимых меток атома
    """
    attach = 1
    insert = 2

_str_to_atom_label = {
    'attach': AtomLabelType.attach,
    'insert': AtomLabelType.insert
}


class RadicalRequirementType(Enum):
    """
    Список типов строгости метки
    """
    fixed = 1
    alternate = 2

_str_to_radical_requirement = {
    'fixed': RadicalRequirementType.fixed,
    'alternate': RadicalRequirementType.alternate
}


class BondLabelType(Enum):
    """
    Список допустимых типов меток связи
    """
    remove = 1

_str_to_bond_label = {
    'remove': BondLabelType.remove
}

_attach_bond_types = {'-', '=', '$', '#'}


"""
Хранит информацию о метке атома attach:
  :param type: AtomLabelType.attach -- тип всегда постоянный
  :param pos: int -- позиция в строке SMILES, на которой встретилась метка
  :param requirement_type: RadicalRequirementType -- тип строгости метки
  :param bonds: tuple -- список свящей для привязывания радикала, 
    используется указанный в метке порядок 
"""
AtomAttachLabel = namedtuple('AtomAttachLabel', ['type', 'pos', 'requirement_type', 'bonds'])

"""
Хранит информацию о метке атома remove:
  :param type: AtomLabelType.remove -- тип всегда постоянный
  :param pos: int -- позиция в строке SMILES, на которой встретилась метка
  :param requirement_type: RadicalRequirementType -- тип строгости метки
"""
AtomInsertLabel = namedtuple('AtomInsertLabel', ['type', 'pos', 'requirement_type'])


"""
Хранит информацию о метке ребра remove:
  :param type: AtomLabelType.attach -- тип всегда постоянный.
  :param pos: int -- позиция в строке SMILES, на которой встретилась метка
  :param count: int -- разрешённое количество удаляемых рёбер
"""
BondRemoveLabel = namedtuple('BondRemoveLabel', ['type', 'pos', 'count'])


def _parse_atom_attach_label(tokens, smiles_pos):
    _assert_token(tokens, Token.colon, 'Ожидается :')
    requirement_type = _parse_token_requirement_type(tokens)
    _assert_token(tokens, Token.colon, 'Ожидается :')
    bonds = _parse_attach_point_bonds(tokens)
    _assert_token(tokens, Token.right_curly_bracket, 'Ожидается }')
    return AtomAttachLabel(
        type=AtomLabelType.attach, 
        pos=smiles_pos,
        requirement_type=requirement_type,
        bonds=bonds
    )

def _parse_atom_insert_label(tokens, smiles_pos):
    _assert_token(tokens, Token.colon, 'Ожидается :')
    requirement_type = _parse_token_requirement_type(tokens)
    _assert_token(tokens, Token.right_curly_bracket, 'Ожидается }')
    return AtomInsertLabel(
        type=AtomLabelType.insert,
        pos=smiles_pos,
        requirement_type=requirement_type
    )

def _parse_bond_remove_label(tokens, smiles_pos):
    _assert_token(tokens, Token.colon, 'Ожидается :')
    cnt_token = _assert_token(tokens, Token.text, 'Ожидается количество удаляемых связей')
    _assert_token(tokens, Token.right_curly_bracket, 'Ожидается {')
    count = int(cnt_token.data)
    return BondRemoveLabel(
        type=BondLabelType.remove,
        pos=smiles_pos,
        count=count
    )


_atom_labels = {
    'attach': _parse_atom_attach_label,
    'insert': _parse_atom_insert_label
}

_bond_labels = {
    'remove': _parse_bond_remove_label
}


def parse(meta_smiles):
    """
    Принимает MetaSMILES строку и возвращает ParseResult

    :raise: MetaSmilesFormatError, если строка содержит ошибки в записи формата MetaSMILES
    """
    tokens = Tokenizer(meta_smiles)
    smiles, atom_labels, bond_labels = [], [], []
    smiles_length = 0
    s = _assert_token(tokens, Token.text, 'Ожидается SMILES')
    smiles.append(s.data)
    smiles_length += len(s.data)

    for token in tokens:
        if token.type == Token.text:
            smiles.append(token.data)
            smiles_length += len(token.data)
        elif token.type == Token.left_curly_bracket:
            label_token = _assert_token(tokens, Token.text, 'Ожидается тип метки')
            label_type = label_token.data
            if label_type in _atom_labels:
                atom_labels.append(_atom_labels[label_type](tokens, smiles_length - 1))
            elif label_type in _bond_labels:
                bond_labels.append(_bond_labels[label_type](tokens, smiles_length - 1))
            else:
                raise MetaSmilesFormatError(
                    Token.text, Token.text, label_token.position, 
                    'Неизвестный тип метки')
        elif token.type != Token.end:
            raise MetaSmilesFormatError(
                Token.text, token.type, token.position, 
                'Ожидается SMILES или {')
    return ParseResult(
        smiles=''.join(smiles),
        atom_labels=atom_labels,
        bond_labels=bond_labels
    )


def _parse_token_requirement_type(tokens):
    """ Разбирает тип строгости меток (attach и insert) атома """
    type_token = _assert_token(tokens, Token.text, 'Ожидается тип строгости метки')
    if type_token.data not in _str_to_radical_requirement:
        msg = 'Неверный тип строгости метки. Указан "{}", допустимы {}'.format(
            type_token.data, ', '.join(_str_to_radical_requirement)
        )
        raise MetaSmilesFormatError(Token.text, type_token.type, type_token.position, msg)
    return _str_to_radical_requirement[type_token.data]


def _parse_attach_point_bonds(tokens):
    """ Разбирает список разрешённых типов связей для метки атома attach """
    bonds_token = _assert_token(tokens, Token.text, 'Ожидаются типы связей для attach')
    for shift, bond in enumerate(bonds_token.data):
        if bond not in _attach_bond_types:
            msg = ('Неверный тип связи для радикала типа attach. '
                   'Используется {}, допустимы {}'.format(bond, ', '.join(_attach_bond_types)))
            raise MetaSmilesFormatError(
                Token.text, bonds_token.type, bonds_token.position + shift, msg)
    return tuple(bonds_token.data)


def _assert_token(tokens, expected_token_type, message):
    """
    :param tokens: итератор по TokenInfo
    :param expected_token_type: тип следующего ожидаемого токена
    :param message: сообщение, если попался другой токен
    :return: ожидаемый токен
    """
    next_token = next(tokens)
    if next_token.type != expected_token_type:
        raise MetaSmilesFormatError(expected_token_type, next_token.type, 
                                    next_token.position, message)
    return next_token
