#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
import sys

if sys.version_info[:2] != (2, 7):
    print('Error: Python 2.7 is required ({}.{} detected).'.format(*sys.version_info[0:2]))
    sys.exit(-1)


import timeit

from gws import generate as g
from gws.io import get_data as gd
from gws.isomorph import graph_kernel as gk


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
    
    Каждая молекула описывается словарем:

    g:         молекулярный граф
    gh:        граф атом-водород
    atom:      список атомов молекулы
    atom_pos:  позиции атомов в SMILES
    hb:        матрица хиральных связей
    sb:        матрица жестких связей
    charge:    матрица заряда
    poia:      список атомов для замены на инсерты
    poih:      список атомов для добавления аттачей
    poia_add:  список атомов для замены на инсерты для следующего поколения молекул
    poih_add:  список атомов для добавления аттачей для следующего поколения молекул
    smiles:    SMILES-строка молекулы
    history:   история добавления аттачей и инсертов

    Для аттачей и инсертов используется дополнительное поле
    name:      имя аттача/инсерта

    Для аттачей используется дополнительное поле
    bound:     число, кратность связи аттача ('-': 1, '=': 2, '#': 3)

    Порядок генерации молекул следующий:
     1. Задать star-SMILES для фрейма (основной молекулы)
     2. Преобразовать star-SMILES в молекулу через data_prep_frame
     3. Задать словарь аддонов, содержащий лист имен инсертов/аттачей ('names') 
          и лист star-SMILES инсертов/аттачей ('smiles')
     4. Преобразовать словарь аддонов в лист молекул-аддонов через data_prep_adds
     5. Выполнить генерацию через 
          list_mols, list_mols_smiles = generate(N, frame, adds),
          где N - максимальное количество аттачей/инсертов для добавления

    Функция generate возвращает два листа. 
      list_mols содержит лист молекул в формате словаря. 
      list_mols_smiles содержит лист сгенерированных SMILES

    Если требуется обработать полученные молекулы, а после довесить 
      к аттачам/инсертам новые аттачи/инсерты, необходимо взять соответсвующую
      молекулу из list_mols и использовать ее как фрейм для дальнейшей генерации. 
      В этом словаре уже содержится необходимая информация для начала добавления 
      аддонов к ранее добавленным. 

      Для продолжения генерации достаточно выполнить пункты 3-5.
    """

    # Исходная молекула ы формате star_smiles
    start_star_smiles = 'C1C*C*C*C*C1C2CC**C**CC2'

    # Набор аддонов
    # TODO ! rename attach -> attaches, insert -> inserts
    addons_data = {'attach': ['-{C=CN}', '-{Cl}'], 'names_at': ['n1', 'cl'],
                   'insert': ['{Cl}', '{O}'], 'names_in': ['CCl', 'O']}

    # Преобразование star-SMILES в молекулу
    frame = gd.data_prep_frame(start_star_smiles)

    # Преобразование аддонов в молекулы
    addons = gd.data_prep_addons(addons_data)


    # Проверка на изоморфизм через GK
    gk_param = gk.get_def_par()
    gk_param['p'] = 0.9999
    n = 3  # Количество добовляемых аттачей
    start_time = timeit.default_timer()

    # Генерация первого поколения молекул
    # к аддонам здесь ничего не добавляется
    list_mols, list_mols_smiles = g.generate(n, frame, addons, gk_param=gk_param)
    print('GK time = {}'.format(timeit.default_timer() - start_time))

    # Честная проверка
    start_time = timeit.default_timer()
    list_mols_a, list_mols_smiles_a = g.generate(n, frame, addons, 1)
    print('Fair isomorphism time = {}'.format(timeit.default_timer() - start_time))
    print(len(list_mols_smiles), len(list_mols_smiles_a))
    print(list_mols_smiles)  # изоморфизм через GK
    print(list_mols_smiles_a)  # изоморфизм честный

    # Выбираем в качестве нового фрейма ранее сгенерированную молекулу
    new_frame_mol = list_mols[-1]

    # Задаем новые аддоны
    addons_data = {'attach': ['-{NC}'], 'names_at': ['n3'], 
                   'insert': ['{NO}', '{COC}'], 'names_in': ['n4', 'n5']}  
    # Преобразовываем аддоны в молекулы
    addons = gd.data_prep_addons(addons_data)

    # Генерируем молекулы второго поколения
    list_mols_second_gen, list_mols_smiles_second_gen = g.generate(2, new_frame_mol, addons)
    print(list_mols_smiles_second_gen)


if __name__ == "__main__":
    main()