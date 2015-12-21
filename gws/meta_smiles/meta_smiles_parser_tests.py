import re
import unittest
import parser

from parser_exceptions import MetaSmilesFormatError

class MetaSmilesParserTests(unittest.TestCase):
    def test_no_marks(self):
        meta_smiles = 'C=C-C(O)NP#PC(O)(O)O'
        result = parser.parse(meta_smiles)
        self.assertEqual(result.smiles, meta_smiles)
        self.assertFalse(result.atom_labels)
        self.assertFalse(result.bond_labels)

    def test_atom_simple_attach_label(self):
        meta_smiles = 'CC{attach:fixed:-}C(O)NP'
        smiles = re.sub('\{.*?\}', '', meta_smiles)
        result = parser.parse(meta_smiles)
        self.assertEqual(result.smiles, smiles)
        self.assertEqual(result.atom_labels, [parser.AtomAttachLabel(
            type=parser.AtomLabelType.attach,
            pos=1,
            requirement_type=parser.RadicalRequirementType.fixed,
            bonds=('-',)
        )])
        self.assertFalse(result.bond_labels)

    def test_atom_simple_insert_label(self):
        meta_smiles = 'CC{insert:fixed}C(O)NP'
        smiles = re.sub('\{.*?\}', '', meta_smiles)
        result = parser.parse(meta_smiles)
        self.assertEqual(result.smiles, smiles)
        self.assertEqual(result.atom_labels, [parser.AtomInsertLabel(
            type=parser.AtomLabelType.insert,
            pos=1,
            requirement_type=parser.RadicalRequirementType.fixed,
        )])
        self.assertFalse(result.bond_labels)

    def test_atom_simple_remove_label(self):
        meta_smiles = 'C1C-{remove:1}C(O)NP1'
        smiles = re.sub('\{.*?\}', '', meta_smiles)
        result = parser.parse(meta_smiles)
        self.assertEqual(result.smiles, smiles)
        self.assertFalse(result.atom_labels)
        self.assertEqual(result.bond_labels, [parser.BondRemoveLabel(
            type=parser.BondLabelType.remove,
            pos=3,
            count=1
        )])

    def test_atom_labels(self):
        meta_smiles = 'CC{insert:fixed}{attach:alternate:-=}C(O)NP'
        smiles = re.sub('\{.*?\}', '', meta_smiles)
        result = parser.parse(meta_smiles)
        self.assertEqual(result.smiles, smiles)
        self.assertEqual(set(result.atom_labels), {
            parser.AtomAttachLabel(
                type=parser.AtomLabelType.attach,
                pos=1,
                requirement_type=parser.RadicalRequirementType.alternate,
                bonds=('-', '=')
            ),
            parser.AtomInsertLabel(
                type=parser.AtomLabelType.insert,
                pos=1,
                requirement_type=parser.RadicalRequirementType.fixed,
            )})
        self.assertFalse(result.bond_labels)

    def test_complicated_labels(self):
        meta_smiles = 'C1C{insert:fixed}(O{attach:fixed:-})-{remove:1}C1'
        smiles = re.sub('\{.*?\}', '', meta_smiles)
        result = parser.parse(meta_smiles)
        self.assertEqual(result.smiles, smiles)
        self.assertEqual(set(result.atom_labels), {
            parser.AtomAttachLabel(
                type=parser.AtomLabelType.attach,
                pos=4,
                requirement_type=parser.RadicalRequirementType.fixed,
                bonds=('-',)
            ),
            parser.AtomInsertLabel(
                type=parser.AtomLabelType.insert,
                pos=2,
                requirement_type=parser.RadicalRequirementType.fixed,
            )
        })
        self.assertEqual(set(result.bond_labels), {
            parser.BondRemoveLabel(
                type=parser.BondLabelType.remove,
                pos=6,
                count=1
            )
        })

    def test_bad_meta_smiles(self):
        meta_smiles = 'CC{attack:fixed:-=}C'
        with self.assertRaises(MetaSmilesFormatError):
            parser.parse(meta_smiles)


if __name__ == '__main__':
    unittest.main()
