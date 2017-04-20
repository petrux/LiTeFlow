"""Vocabularies for managing symbols encoding/decoding.

In this module, a base class `BaseVocabulary` defines the main API for a vocabulary
that maps words to integer values and viceversa. Such poblem is very common dealing
with natural language processing tasks, so a shared API and some already implemented
code can be handy sometimes.
"""

import abc


class BaseVocabulary(object):
    """Base vocabulary read-only interface.

    The `BaseVocabulary` abstract class provides the base interface
    for a vocabulary containing a set of |V| words and their corresponding
    index, an integer value ranging from 0 to |V|-1.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def contains(self, word):
        """Check if a word is contained in the vocabulary.

        Arguments:
          word: a word from the vocabulary.

        Returns:
          `True` if the word is contained in the vocabulary,
            `False` otherwise.
        """
        raise NotImplementedError(
            """The abstract method `contains(self, word)` """
            """must be implemented in subclasses.""")

    @abc.abstractmethod
    def index(self, word):
        """Get the index value for the given word.

        Arguments:
          word: a word from the vocabulary.

        Returns:
          an `int` representing the index value of the
            given word.

        Raises:
          ValueError: if the word is not in the vocabulary.
        """
        raise NotImplementedError(
            """The abstract method `index(self, word)` """
            """must be implemented in subclasses.""")

    @abc.abstractmethod
    def word(self, index):
        """Get the word for the given index.

        Arguments:
          index: an `int` representing an index value for a word.

        Returns:
          the word corresponding to the given index value.

        Raises:
          ValueError: if the index value is not between 0 and the
            number of words contained in the vocabulary minus 1.
        """
        raise NotImplementedError(
            """The abstract method `word(self, index)` """
            """must be implemented in subclasses.""")

    @abc.abstractmethod
    def size(self):
        """Get the number of words in the vocabulary."""
        raise NotImplementedError(
            """The abstract method `size(self)` """
            """must be implemented in subclasses.""")

    @abc.abstractmethod
    def items(self):
        """Return an iterator over the pairs (index, word)."""
        raise NotImplementedError(
            """The abstract method `items(self)` """
            """must be implemented in subclasses.""")

    def __contains__(self, item):
        """Magic method wrapping the `contains()` method."""
        return self.contains(item)

    def __len__(self):
        """Magic method wrapping the `size()` method."""
        return self.size()

    def __iter__(self):
        """Magic method wrapping the `items()` method."""
        return self.items()


class InMemoryVocabulary(BaseVocabulary):
    """In-memory implementation of the BaseVocabulary contract.

    The InMemoryVocabulary class holds in-memory data structure to
    extend the BaseVocabulary superclass. All the access operations
    are ensured to have O(1) time complexity.
    """

    def __init__(self):
        self._index = {}
        self._words = []

    def contains(self, word):
        return word in self._index

    def index(self, word):
        if word in self._index:
            return self._index[word]
        raise ValueError('Word \'%s\' is not in the vocabulary.' % word)

    def word(self, index):
        if index < 0 or index >= len(self._words):
            raise ValueError('Index must be between 0 and %d, found %d instead'
                             % (len(self._words) - 1, index))
        return self._words[index]

    def size(self):
        return len(self._words)

    def items(self):
        return enumerate(self._words)

    def add(self, word):
        """Add a new word to the vocabulary.

        Arguments:
          word: a word to be added to the vocabulary.

        Returns:
          The index of the word.

        Remarks:
          if the word is already in the vocabulary, it is not added twice
            and its current index value will be returned.
        """

        if word in self._words:
            return self._index[word]

        index = len(self._words)
        self._words.append(word)
        self._index[word] = index
        return index
    