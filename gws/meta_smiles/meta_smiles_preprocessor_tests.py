import unittest
from preprocessor import preprocess

class MetaSmilesPreprocessorTests(unittest.TestCase):
    def test_no_preprocess(self):
        s = 'test{abc}test2{bcd}test3'
        result = list(preprocess(s))
        expected = [s]
        self.assertEqual(result, expected)

    def test_basic_string(self):
        s = '{a|b}'
        result = set(preprocess(s))
        self.assertEqual(result, {'{a}', '{b}'})

    def test_two_long_parts(self):
        parts= {'asdfg', 'qwertyuiop'}
        s = '{' + '|'.join(parts) + '}'
        result = set(preprocess(s))
        expected = set('{' + value + '}' for value in parts)
        self.assertEqual(expected, result)

    def test_three_part_in_one_group(self):
        parts= {'qwerty', 'asdfg', 'zxcvb'}
        s = '{' + '|'.join(parts) + '}'
        result = set(preprocess(s))
        expected = set('{' + value + '}' for value in parts)
        self.assertEqual(expected, result)

    def test_preprocess_with_prefix_and_suffix(self):
        parts = {'qwe', 'wer'}
        prefix, suffix = 'prefix', 'suffix'
        s = ''.join((prefix, '{', '|'.join(parts), '}', suffix))
        result = set(preprocess(s))
        expected = set(''.join((prefix, '{', value, '}', suffix)) 
                       for value in parts)
        self.assertEqual(expected, result)

    def test_tree_mult_blocks(self):
        parts1 = {'g1.1', 'g1.2', 'g1.3', 'g1.4'}
        parts2 = {'g2.1', 'g2.2', 'g2.3', 'g2.4'}
        parts3 = {'g3.1', 'g3.2', 'g3.3', 'g3.4'}
        prefix, middle1, middle2, suffix = 'prefix', 'middle1', 'middle2', 'suffix'
        s = ''.join((prefix, '{', '|'.join(parts1), '}', middle1, 
                    '{', '|'.join(parts2), '}', middle2, 
                    '{', '|'.join(parts3), '}', suffix))
        result = set(preprocess(s))
        expected = set(''.join((prefix, '{', value1, '}', middle1, 
                                '{', value2, '}', middle2, 
                                '{', value3, '}', suffix)) 
                       for value1 in parts1 for value2 in parts2 for value3 in parts3)
        self.assertEqual(expected, result)

    def test_trash_symbols_no_preprocess(self):
        s = ('`1234567890-=qwertyuiop[]asdfghjkl;\'zxcvbnm,./~!@#$%^&*()_+QWERT'
             'YUIOPASDFGHJKL:"ZXCVBNM<>?\\')
        result = list(preprocess(s))
        self.assertEqual(result, [s])


if __name__ == '__main__':
    unittest.main()