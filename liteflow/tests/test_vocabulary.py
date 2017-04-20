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
        self.assertEquals(0, contains.call_count)

        arg = 23
        _ = arg in vocab
        self.assertEquals(1, contains.call_count)
        contains.assert_called_with(arg)

        arg = object()
        _ = arg in vocab
        self.assertEquals(2, contains.call_count)
        contains.assert_called_with(arg)

    @mock.patch.object(_BaseVocabulary, 'size')
    def test_size(self, size):
        """Test that __len__ bounces on size()."""

        vocab = _BaseVocabulary()
        self.assertEquals(0, size.call_count)

        _ = len(vocab)
        self.assertEquals(1, size.call_count)

        _ = len(vocab)
        self.assertEquals(2, size.call_count)

    @mock.patch.object(_BaseVocabulary, 'items')
    def test_items(self, items):
        """Test that __iter__ bounces on items()."""

        vocab = _BaseVocabulary()
        items.return_value = iter([])
        self.assertEquals(0, items.call_count)
        _ = iter(vocab)
        self.assertEquals(1, items.call_count)


class InMemoryVocabularyTest(unittest.TestCase):
    """Test case for the `liteflow.vocabulary.InMemoryVocabulary` class."""

    def test_empty(self):
        """Test the empty vocabulary."""
        vocab = vocabulary.InMemoryVocabulary()
        self.assertEquals(0, len(vocab))

    def test_base(self):
        """Test the basic functionalities of the vocabulary."""

        words = 'A B C X Y Z'.split()
        vocab = vocabulary.InMemoryVocabulary()

        for i, word in enumerate(words):
            self.assertFalse(word in vocab)
            self.assertEquals(i, vocab.add(word))
            self.assertTrue(word in vocab)
            self.assertEquals(i, vocab.index(word))
            self.assertEquals(word, vocab.word(i))
            self.assertEquals(i + 1, len(vocab))

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
            self.assertEquals(word, vocab.word(index))
        self.assertRaises(ValueError, lambda: vocab.word(-1))
        self.assertRaises(ValueError, lambda: vocab.word(len(words)))

    def test_add_twice(self):
        """Test adding a word twice."""

        word = 'WORD'
        vocab = vocabulary.InMemoryVocabulary()
        index = vocab.add(word)
        self.assertEquals(1, vocab.size())
        self.assertEquals(index, vocab.add(word))
        self.assertEquals(1, vocab.size())


if __name__ == '__main__':
    unittest.main()
