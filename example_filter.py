#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
import sys

import timeit
import numpy as np
from rdkit import Chem

from gws import generate as g
from gws.io import get_data as gd
from gws.isomorph import graph_kernel as gk
from gws import molecule_from_star_smiles

if sys.version_info[:2] != (2, 7):
    print('Error: Python 2.7 is required ({}.{} detected).'.format(*sys.version_info[0:2]))
    sys.exit(-1)


class FilterExample(object):
    """
    В каждом фильтре должна быть обязательная переменная self.is_filter и метод get_score
    self.is_filter определяет тип использования фильтра: если self.is_filter = True, то GWS ожидает
    на выходе get_score list из bool переменных. Используются только те молекулы, которые имеют
    метку True. self.is_filter = False, то GWS ожидает на выходе числовую характеристику молекулы,
    и чем она меньше, тем лучше молекула. Если self.is_filter = False, то должен быть определен
    параметр self.max_mols: на каждой итерации GWS выбирается не более self.max_mols молекул.
    """
    def __init__(self, patterns):
        self.is_filter = True
        self.patterns = patterns

    def get_score(self, smiles, rdkit_mols):
        output = []
        for m in rdkit_mols:
            is_ok = True
            for pattern in self.patterns:
                if m.HasSubstructMatch(pattern):
                    is_ok = False
                    break
            output.append(is_ok)
        return output


class NotFilterExample(object):
    def __init__(self, max_mols):
        self.is_filter = False
        self.max_mols = max_mols

    def get_score(self, smiles, rdkit_mols):
        output = []
        for _ in rdkit_mols:
            output.append(np.random.ranf())
        return output


def main():
    """
    В данном примере продемонстрирована последовательность
      генерации малых молекул

    Используемая терминология:
     - frame, фрейм: основная молекула, к которой применяются изменения,
         на основе которой генерируется конечное множество молекул;
     - attach, аттач: молекула, прицепляемая к атому фрейма, обычно заменяющая
         собой водородный атом;
     - insert, инсерт: молекула, предназначенная для замены атома фрейма;
     - addon, аддон: аттач или инсерт

    Для аддонов используется дополнительное поле
    name:      имя аддона

    Для аттачей используется дополнительное поле
    bound:     число, кратность связи аттача ('-': 1, '=': 2, '#': 3)

    Порядок генерации молекул следующий:
     1. Задать star-smiles для фрейма
     2. Преобразовать star-smiles в молекулу через star_smiles_to_mol
     3. Задать словарь аддонов, содержащий лист имен аддонов ('names')
          и лист star-smiles аддонов ('smiles')
     4. Преобразовать словарь аддонов в лист молекул-аддонов через data_prep_adds
     5. Выполнить генерацию через
          list_mols, list_mols_smiles = generate(N, frame, adds),
          где N - максимальное количество аддонов для добавления

    Функция generate возвращает два листа.
      list_mols содержит лист молекул в формате словаря.
      list_mols_smiles содержит лист сгенерированных SMILES

    Если требуется обработать полученные молекулы, а после довесить
      к аддонам новые, необходимо взять соответствующую молекулу из list_mols
      и использовать ее как фрейм для дальнейшей генерации.
      В этом словаре уже содержится необходимая информация для начала добавления
      аддонов к ранее добавленным.

      Для продолжения генерации достаточно выполнить пункты 3-5.
    """

    # Исходная молекула ы формате star_smiles
    start_star_smiles = 'C1C*C*C*C*C1C2CC**C**CC2CC*N'

    # Набор аддонов
    addons_data = {'attach': ['{N<#>C<=>(Cl)<->C<=>}', '{N<#>C<=>(Cl)<->N}', '{S<=>}'],
                   'names_at': ['n1', 'cl', 's'],
                   'insert': ['{Cl}', '{O}'], 'names_in': ['CCl', 'O']}

    # Преобразование star-smiles в молекулу
    frame = molecule_from_star_smiles(start_star_smiles)

    # Преобразование аддонов в молекулы
    addons = gd.data_prep_addons(addons_data)

    print('Test GWS with filter score')
    patterns = ['[R0;D2][R0;D2][R0;D2][R0;D2]', '[CR0]=[CR0][CR0]=[CR0]', 'CC(=S)N']
    alerts = []
    for smrt in patterns:
        m = Chem.MolFromSmarts(smrt)
        if m is not None:
            alerts.append(m)

    score_filter = FilterExample(alerts)

    # Проверка на изоморфизм через GK
    gk_param = gk.get_def_par()
    gk_param['p'] = 0.9999
    n = 1  # Количество добовляемых аттачей
    start_time = timeit.default_timer()

    # Генерация первого поколения молекул
    # к аддонам здесь ничего не добавляется
    list_mols, list_mols_smiles = g.generate(n, frame, addons, gk_param=gk_param,
                                             score=score_filter)
    print('GK time = {}'.format(timeit.default_timer() - start_time))

    print(len(list_mols_smiles))
    print(list_mols_smiles)

    print('='*30)
    print('Test GWS with not filter score')

    score_filter = NotFilterExample(5)

    # Проверка на изоморфизм через GK
    gk_param = gk.get_def_par()
    gk_param['p'] = 0.9999
    n = 1  # Количество добовляемых аттачей
    start_time = timeit.default_timer()

    # Генерация первого поколения молекул
    # к аддонам здесь ничего не добавляется
    list_mols, list_mols_smiles = g.generate(n, frame, addons, gk_param=gk_param,
                                             score=score_filter)
    print('GK time = {}'.format(timeit.default_timer() - start_time))

    print(len(list_mols_smiles))
    print(list_mols_smiles)

if __name__ == "__main__":
    main()
