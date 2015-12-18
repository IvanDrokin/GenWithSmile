# coding=utf-8

from io.get_data import star_smiles_to_mol


def molecule_from_star_smiles(star_smiles):
    star_smiles = star_smiles
    raw_mol = star_smiles_to_mol(star_smiles)
    return Molecule(raw_mol)


class Molecule:
    def __init__(self, raw_mol):
        self._raw_mol = raw_mol
        self.poia = self._raw_mol['poia']
        self.poih = self._raw_mol['poih']
        self.poif = self._raw_mol['poif']
        self.poia_add = self._raw_mol['poia_add']
        self.poih_add = self._raw_mol['poih_add']
        self.smiles = self._raw_mol['smiles']
        self.history = self._raw_mol['history']
        self.graph_kernel_vect = None
        self.rdkit_mol = self._raw_mol['rdkit_mol']
        self.attach_index = None
        self.bound = None
        self.name = None
        if 'attach_index' in raw_mol:
            self.attach_index = raw_mol['attach_index']
        if 'bound' in raw_mol:
            self.attach_index = raw_mol['bound']
        if 'name' in raw_mol:
            self.attach_index = raw_mol['name']
