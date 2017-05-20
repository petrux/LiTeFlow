"""Test module for the `liteflow.vocabulary` module."""

import unittest

import mock

from liteflow import vocabulary


class _BaseVocabulary(vocabulary.BaseVocabulary):
    def contains(self, word): pass  # pylint: disable=I0011,C0321
    def index(self, word): pass  # pylint: disable=I0011,C0321
    def word(self, index): pass  # pylint: disable=I0011,C0321
    def size(self): pass  # pylint: disable=I0011,C0321
    def items(self): pass  # pylint: disable=I0011,C0321


class BaseVocabularyTest(unittest.TestCase):
    """Test case for the `liteflow.vocabulary.BaseVocabulary` contract."""

    @mock.patch.object(_BaseVocabulary, 'contains')
    def test_contains(self, contains):
        """Test that __contains__ bounces on contains()."""

        vocab = _BaseVocabulary()
        self.assertEqual(0, contains.call_count)

        arg = 23
        _ = arg in vocab
        self.assertEqual(1, contains.call_count)
        contains.assert_called_with(arg)

        arg = object()
        _ = arg in vocab
        self.assertEqual(2, contains.call_count)
        contains.assert_called_with(arg)

    @mock.patch.object(_BaseVocabulary, 'size')
    def test_size(self, size):
        """Test that __len__ bounces on size()."""

        vocab = _BaseVocabulary()
        self.assertEqual(0, size.call_count)

        _ = len(vocab)
        self.assertEqual(1, size.call_count)

        _ = len(vocab)
        self.assertEqual(2, size.call_count)

    @mock.patch.object(_BaseVocabulary, 'items')
    def test_items(self, items):
        """Test that __iter__ bounces on items()."""

        vocab = _BaseVocabulary()
        items.return_value = iter([])
        self.assertEqual(0, items.call_count)
        _ = iter(vocab)
        self.assertEqual(1, items.call_count)


class InMemoryVocabularyTest(unittest.TestCase):
    """Test case for the `liteflow.vocabulary.InMemoryVocabulary` class."""

    def test_empty(self):
        """Test the empty vocabulary."""
        vocab = vocabulary.InMemoryVocabulary()
        self.assertEqual(0, len(vocab))

    def test_base(self):
        """Test the basic functionalities of the vocabulary."""

        words = 'A B C X Y Z'.split()
        vocab = vocabulary.InMemoryVocabulary()

        for i, word in enumerate(words):
            self.assertFalse(word in vocab)
            self.assertEqual(i, vocab.add(word))
            self.assertTrue(word in vocab)
            self.assertEqual(i, vocab.index(word))
            self.assertEqual(word, vocab.word(i))
            self.assertEqual(i + 1, len(vocab))

    def test_oov_words(self):
        """Test out-of-vocabulary words."""

        unk = 'Q'
        words = 'A B C X Y Z'.split()
        vocab = vocabulary.InMemoryVocabulary()
        for word in words:
            vocab.add(word)

        self.assertFalse(unk in vocab)
        self.assertRaises(ValueError, lambda: vocab.index(unk))

    def test_oov_indexes(self):
        """Test out-of-vocabulary indexes."""

        words = 'A B C X Y Z'.split()
        vocab = vocabulary.InMemoryVocabulary()
        for word in words:
            vocab.add(word)

        for index, word in enumerate(words):
            self.assertEqual(word, vocab.word(index))
        self.assertRaises(ValueError, lambda: vocab.word(-1))
        self.assertRaises(ValueError, lambda: vocab.word(len(words)))

    def test_add_twice(self):
        """Test adding a word twice."""

        word = 'WORD'
        vocab = vocabulary.InMemoryVocabulary()
        index = vocab.add(word)
        self.assertEqual(1, vocab.size())
        self.assertEqual(index, vocab.add(word))
        self.assertEqual(1, vocab.size())


class TestUNKVocabulary(unittest.TestCase):
    """Test case for the `liteflow.vocabulary.UNKVocabulary` class."""

    def test_unknown(self):
        """Test the support for unknwon words."""
        unk = vocabulary.UNKVocabulary.UNK
        voc = vocabulary.InMemoryVocabulary()
        voc.add(unk)
        voc.add('A')
        voc.add('B')
        voc.add('C')
        unkvoc = vocabulary.UNKVocabulary(voc)

        word = 'X'
        self.assertEqual(4, unkvoc.size())
        self.assertFalse(word in unkvoc)
        index = unkvoc.index(word)
        self.assertEqual(unk, unkvoc.word(index))
        self.assertEqual(index, unkvoc.index(unk))

if __name__ == '__main__':
    unittest.main()
