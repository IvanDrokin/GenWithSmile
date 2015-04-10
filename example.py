# coding=utf-8
import os
import sys

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


def prepare():
    if sys.version_info[:2] != (2, 7):
        print("Error: Python 2.7 is required (%d.%d detected)." % (sys.version_info[0], sys.version_info[1]))
        sys.exit(-1)
    sys.path.insert(0, ROOT_PATH)


def run():
    '''
    В данном примере продемонстрирована последовательность генерации малых молекул
    Каждая молекула описывается словарем

    'g': g = граф молекулярный граф
    'gh': gh = граф атом-водород
    'atom': a = список атомов молекулы
    'atom_pos': ap = позиции атомов в SMILES
    'hb': hb = матрица хиральных связей
    'sb': sb = матрица жестких связей
    'charge': chaarge = матрица заряда
    'poia': poia = список атомов для замены на инсерты
    'poih': poih = список атомов для добавления аттачей
    'poia_add': poia_add = список атомов для замены на инсерты для следующего поколения молекул
    'poih_add': poih_add = список атомов для добавления аттачей для следующего поколения молекул
    'smiles': sm = SMILES
    'history': hist = История добавления аттачей и инсертов

    Для аттачей и инсертов используется дополнительное поле
    'name' = имя аттача/инсерта
    И для аттачей используется поле
    'bound' = тип связи для аттачи ('-', '=', '#')

    Порядок генерации молекул следующий:
    1. Задать star-SMILES для фрейма
    2. Преобразовать star-SMILES в молекулу через data_prep_frame
    3. Задать словарь аддонов, содержащий лист имен инсертов/аттачей ('names') и лист star-SMILES инсертов/аттачей ('smiles')
    4. Преобразовать словарь аддонов в лист молекул-аддонов через data_prep_adds
    5. Выполнить генерацию через list_mols, list_mols_smiles = generate(N, frame, adds),
            где N - максимальное количество аттачей/инсертов для добавления

    Функция generate возвращает два листа. list_mols содержит лист молекул в формате словаря. list_mols_smiles содержит лист
    сгенерированных SMILES
    Если требуется обработать полученные молекулы, а после довесить к аттачам/инсертам новые аттачи/инсерты,
    необходимо взять соответсвующую молекулу из list_mols и использовать ее как фрейм для дальнейшей генерации. В этом
    словаре уже содержится необходимая информация для начала добавления аддонов к ранее добавленным. Для продолжения
    генерации достаточно выполнить пункты 3-5.
    '''
    frame_smiles = 'Cl1C*C*C*C*C1C2CC**C**CC2'  # исходный фрейм с точками для аттача
    adds_smiles = {'attach': ['-{C=CN}', '-{Cl}'], 'names_at': ['n1', 'cl'],
                   'insert': ['{Cl}', '{O}'], 'names_in': ['CCl', 'O']}  # Набор аттачей

    frame = gd.data_prep_frame(frame_smiles)  # Преобразование star-SMILES в молекулу
    adds = gd.data_prep_adds(adds_smiles)  # преобразование аддонов в молекулы

    gk_param = gk.get_def_par()
    gk_param['p'] = 0.9999
    # Проверка на изоморфизм через GK
    list_mols, list_mols_smiles = g.generate(3, frame, adds, gk_param=gk_param)  # Генерация первого поколения молекул.
                                                                #  К аттачам/инсертам здесь ничего не добавляется
    # Честная проверка
    list_mols_a, list_mols_smiles_a = g.generate(3, frame, adds, 1)

    print len(list_mols_smiles), len(list_mols_smiles_a)
    print (list_mols_smiles)  # изоморфизм через GK
    print (list_mols_smiles_a)  # изоморфизм честный

    new_frame_mol = list_mols[-1]  # Выбираем в качестве нового фрейма ранее сгенерированную молекулу
    adds_smiles = {'attach': ['-{NC}'], 'names_at': ['n3'], 'insert': ['{NO}'], 'names_in': ['n4']}  # Задаем новые аттачи
    adds = gd.data_prep_adds(adds_smiles)  # Преобразовываем аттачи в молекулы

    list_mols_second_gen, list_mols_smiles_second_gen = g.generate(2, new_frame_mol, adds)  # Генерируем молекулы второго поколения

    print list_mols_smiles
    print list_mols_smiles_second_gen


def main():
    run()


if __name__ == "__main__":
    prepare()

    from gws import generate as g
    from gws.io import get_data as gd
    from gws.isomorph import graph_kernel as gk
    main()