from typing import List, Union
from itertools import chain
from collections import Counter
from nltk.util import ngrams
import pandas as pd

from logger import setup_logger

logger = setup_logger(__name__)

class PrefixTreeNode:
    def __init__(self):
        self.children: dict[str, "PrefixTreeNode"] = {}
        self.is_end_of_word = False

class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        logger.debug("Building PrefixTree with vocab size=%d", len(vocabulary))
        self.root = PrefixTreeNode()
        for word in vocabulary:
            current = self.root
            for symbol in word:
                if symbol not in current.children:
                    current.children[symbol] = PrefixTreeNode()
                current = current.children[symbol]
            current.is_end_of_word = True

    def search_prefix(self, prefix: str) -> List[str]:
        logger.debug("Searching prefix: %s", prefix)
        words: List[str] = []
        current = self.root
        for char in prefix:
            if char not in current.children:
                return words
            current = current.children[char]

        def _dfs(node: "PrefixTreeNode", path: list[str]):
            if node.is_end_of_word:
                words.append("".join(path))
            for c, child in node.children.items():
                _dfs(child, path + [c])

        _dfs(current, list(prefix))
        logger.debug("Found %d words for prefix %s", len(words), prefix)
        return words

class WordCompletor:
    def __init__(self, corpus: list[list[str]]):
        logger.info("Initializing WordCompletor with %d sequences", len(corpus))
        vocabulary = list(chain.from_iterable(corpus))
        all_words = max(1, len(vocabulary))
        count = Counter(vocabulary)
        self.words_probs = {key: value / all_words for key, value in count.items()}
        u_words = list(count.keys()) #побыстрее чем set
        self.prefix_tree = PrefixTree(u_words)

    def get_words_and_probs(self, prefix: str) -> tuple[List[str], List[float]]:
        logger.debug("WordCompletor.get_words_and_probs(%s)", prefix)
        words = self.prefix_tree.search_prefix(prefix)
        probs = [self.words_probs[w] for w in words if w in self.words_probs]
        pairs = list(zip(words, probs))
        pairs.sort(key=lambda x: (-x[1], x[0]))
        words_sorted = [w for w, _ in pairs]
        probs_sorted = [p for _, p in pairs]
        logger.debug("Candidates: %s", words_sorted[:5])
        return words_sorted, probs_sorted

class NGramLanguageModel:
    def __init__(self, corpus: list[list[str]], n: int):
        logger.info("Initializing NGramLanguageModel with n=%d", n)
        self.n = max(1, n)
        self.ngrams = list(chain.from_iterable(ngrams(seq, self.n + 1) for seq in corpus))

    def get_next_words_and_probs(self, prefix: list[str]) -> tuple[List[str], List[float]]:
        logger.debug("NGram.get_next_words_and_probs prefix=%s", prefix)
        endings: list[str] = []
        pref_join = " ".join(prefix)
        for ngram_item in self.ngrams:
            if " ".join(ngram_item[:self.n]) == pref_join:
                endings.append(ngram_item[self.n])

        if not endings:
            return [], []

        counts = Counter(endings)
        total = sum(counts.values())
        next_words, probs = zip(*sorted(
            ((w, c / total) for w, c in counts.items()),
            key=lambda x: (-x[1], x[0]) #везде сортировку еще добавила по вероятности
        ))
        next_words = list(next_words)
        probs = list(probs)
        logger.debug("Next candidates: %s", next_words[:5])
        return next_words, probs

class TextSuggestion:
    def __init__(self, word_completor: WordCompletor, n_gram_model: NGramLanguageModel):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model

    def suggest_text(self, text: Union[str, list[str]], n_words=3, n_texts=1) -> list[list[str]]:
        logger.info("Suggesting text for input=%s", text)
        tokens = text.strip().split() if isinstance(text, str) else list(text)
        if not tokens:
            return [[]]

        suggestions: list[list[str]] = []
        for _ in range(max(1, n_texts)):
            current_word = tokens[-1]
            context = tokens[:-1] if len(tokens) > 1 else []

            words, _ = self.word_completor.get_words_and_probs(current_word)
            if not words:
                words = [current_word]

            recommended_word = words[0]
            current_suggestion = [recommended_word]
            current_context = context + [recommended_word]

            for _i in range(max(0, n_words)):
                prefix = (
                    current_context[-self.n_gram_model.n:]
                    if len(current_context) >= self.n_gram_model.n
                    else current_context
                )
                next_words, _probs = self.n_gram_model.get_next_words_and_probs(prefix)
                if not next_words:
                    break
                most_probable_word = next_words[0]
                current_suggestion.append(most_probable_word)
                current_context.append(most_probable_word)

            suggestions.append(current_suggestion)

        logger.info("Suggestions: %s", suggestions)
        return suggestions
