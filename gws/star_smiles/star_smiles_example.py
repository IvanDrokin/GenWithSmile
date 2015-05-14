# coding=utf-8

from parser_exceptions import StarSmilesFormatError
from star_smiles_parser import StarSmilesParser


def star_smiles_to_smiles(star_smiles):
    try:
        parser = StarSmilesParser(star_smiles)
        print(parser.smiles)
        print(parser.attach_positions)
        print(parser.insert_positions)
        print(parser.attach_bonds)
    except StopIteration:
        print('Ошибка в star-smiles. Неожиданный конец строки')
    except StarSmilesFormatError as e:
        print(e)
    return star_smiles


def main():
    print(star_smiles_to_smiles('{CCC}C*CC**<#>C<=>***C{CC<->C}{C}*C<->{CC}**C*'))


if __name__ == '__main__':
    main()