import unittest
from tokens import Token
from tokenizer import Tokenizer, TokenInfo


class MetaSmilesTokenizerTests(unittest.TestCase):
    def test_smiles_without_meta(self):
        s = 'C1CCCC1N(C(=O)O)C1CCCC1'
        result = list(Tokenizer(s))
        expected = [
            TokenInfo(Token.text, 0, s),
            TokenInfo(Token.end, len(s), None)
        ]
        self.assertEqual(result, expected)


    def test_atom_with_group(self):
        s = 'C{type:value:modifier}'
        result = list(Tokenizer(s))
        expected = [
            TokenInfo(Token.text, 0, 'C'),
            TokenInfo(Token.left_curly_bracket, 1, None),
            TokenInfo(Token.text, 2, 'type'),
            TokenInfo(Token.colon, 6, None),
            TokenInfo(Token.text, 7, 'value'),
            TokenInfo(Token.colon, 12, None),
            TokenInfo(Token.text, 13, 'modifier'),
            TokenInfo(Token.right_curly_bracket, 21, None),
            TokenInfo(Token.end, 22, None)
        ]
        self.assertEqual(result, expected)

    def test_complicated_meta_smiles(self):
        s = 'CC{label1}={label2:value1}C(O)N'
        result = list(Tokenizer(s))
        expected = [
            TokenInfo(Token.text, 0, 'CC'),
            TokenInfo(Token.left_curly_bracket, 2, None),
            TokenInfo(Token.text, 3, 'label1'),
            TokenInfo(Token.right_curly_bracket, 9, None),
            TokenInfo(Token.text, 10, '='),
            TokenInfo(Token.left_curly_bracket, 11, None),
            TokenInfo(Token.text, 12, 'label2'),
            TokenInfo(Token.colon, 18, None),
            TokenInfo(Token.text, 19, 'value1'),
            TokenInfo(Token.right_curly_bracket, 25, None),
            TokenInfo(Token.text, 26, 'C(O)N'),
            TokenInfo(Token.end, 31, None)
        ]
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()