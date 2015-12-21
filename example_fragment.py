#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
import sys


import timeit

from gws import generate as g
from gws.io import get_data as gd
from gws.isomorph import graph_kernel as gk
from gws import molecule_from_star_smiles

if sys.version_info[:2] != (2, 7):
    print('Error: Python 2.7 is required ({}.{} detected).'.format(*sys.version_info[0:2]))
    sys.exit(-1)


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
     - fragment, фрагмент: молекула, вставляемая заместо атома в фрейме

    Для аддонов используется дополнительное поле
    name:      имя аддона

    Для аттачей используется дополнительное поле
    bound:                 число, кратность связи аттача ('-': 1, '=': 2, '#': 3)
    two_points:            Тип аттача. Если False, то одноточечный, если True, то двуточечный. Поле
                           необязательное, можно опустить, в таком случае все аттачи будут
                           одноточечными.

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

    Замена атома во фрейме на фрагмент выполняется полным перебором всевозможных вариантов
    присоединения связей, идущих к заменяемому атому, к фрагменту.
    Пример фрагмента: {'fragment': ['{c1ccccc1}'], 'name_fr': ['ar']}
    """

    # Исходная молекула ы формате star_smiles
    start_star_smiles = 'O=CC^NCO'

    # Набор аддонов
    addons_data = {'fragment': ['{c1ccccc1}'], 'name_fr': ['ar']}

    # Преобразование star-smiles в молекулу
    frame = molecule_from_star_smiles(start_star_smiles)

    # Преобразование аддонов в молекулы
    addons = gd.data_prep_addons(addons_data)

    # Проверка на изоморфизм через GK
    gk_param = gk.get_def_par()
    gk_param['p'] = 0.9999
    n = 1  # Количество добовляемых аттачей
    start_time = timeit.default_timer()

    # Генерация первого поколения молекул
    # к аддонам здесь ничего не добавляется
    list_mols, list_mols_smiles = g.generate(n, frame, addons, gk_param=gk_param)
    print('GK time = {}'.format(timeit.default_timer() - start_time))

    print(len(list_mols_smiles))
    print(list_mols_smiles)


if __name__ == "__main__":
    main()
