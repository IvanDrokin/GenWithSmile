from enum import Enum


class BondMark(Enum):
    none = ''
    single = '-'
    double = '='
    triple = '#'


_symbol_to_bond_mark = {
    '-': BondMark.single,
    '=': BondMark.double,
    '#': BondMark.triple
}


def is_bond_symbol(symbol):
    return symbol in _symbol_to_bond_mark


def symbol_to_bond_mark(symbol):
    return _symbol_to_bond_mark.get(symbol, BondMark.none)