# coding=utf-8

from io.get_data import star_smiles_to_mol


def molecule_from_star_smiles(star_smiles):
    star_smiles = star_smiles
    raw_mol = star_smiles_to_mol(star_smiles)
    return Molecule(raw_mol)


class Molecule:
    def __init__(self, raw_mol):
        self._raw_mol = raw_mol
        self.g = self._raw_mol['g']
        self.gh = self._raw_mol['gh']
        self.atom = self._raw_mol['atom']
        self.atom_pos = self._raw_mol['atom_pos']
        self.hb = self._raw_mol['hb']
        self.sb = self._raw_mol['sb']
        self.charge = self._raw_mol['charge']
        self.poia = self._raw_mol['poia']
        self.poih = self._raw_mol['poih']
        self.poia_add = self._raw_mol['poia_add']
        self.poih_add = self._raw_mol['poih_add']
        self.smiles = self._raw_mol['smiles']
        self.history = self._raw_mol['history']