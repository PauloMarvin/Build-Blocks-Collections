import re
from typing import Any, Dict, List
import spacy
from spacy.matcher import Matcher
import nltk


class TextProcessor:
    nlp_object = spacy.load("pt_core_news_lg", disable=["parser", "ner", "tagger"])
    nlp_object.add_pipe("emoji", first=True)

    @classmethod
    def remove_sites_corpus(cls, corpus: List[str]) -> List[str]:
        """Remove URLs do corpus."""
        return [cls.remove_sites(text) for text in corpus]

    @classmethod
    def remove_punctuation_corpus(cls, corpus: List[str]) -> List[str]:
        """Remove pontuações do corpus."""
        return [cls.remove_punctuation(text) for text in corpus]

    @classmethod
    def lower_text_corpus(cls, corpus: List[str]) -> List[str]:
        """Converte texto para minúsculas no corpus."""
        return [cls.lower_text(text) for text in corpus]

    @classmethod
    def remove_stopwords_corpus(cls, corpus: List[str]) -> List[str]:
        """Remove stopwords do corpus."""
        return [cls.remove_stopwords(text) for text in corpus]

    @classmethod
    def lemmatization_corpus(cls, corpus: List[str]) -> List[str]:
        """Realiza lematização no corpus."""
        return [cls.lemmatization(text) for text in corpus]

    @classmethod
    def steaming_corpus(cls, corpus: List[str]) -> List[str]:
        """Realiza stemming no corpus."""
        return [cls.stemming(text) for text in corpus]

    @classmethod
    def remove_with_regex_corpus(
        cls, corpus: List[str], regex_patterns: List[str]
    ) -> List[str]:
        """Remove padrões regex do corpus."""
        return [cls.remove_with_regex(text, regex_patterns) for text in corpus]

    @classmethod
    def remove_with_prefixes_corpus(
        cls, corpus: List[str], prefixes: List[str]
    ) -> List[str]:
        """Remove palavras que começam com prefixos especificados."""
        return [cls.remove_with_prefixes(text, prefixes) for text in corpus]

    @classmethod
    def remove_emojis_corpus(cls, corpus: List[str]) -> List[str]:
        """Remove emojis do corpus."""
        return [cls.remove_emojis(text) for text in corpus]

    @classmethod
    def replace_matches_corpus(
        cls, corpus: List[str], patterns_dict: Dict[str, List[List[Dict[str, Any]]]]
    ) -> List[str]:
        """Substitui padrões definidos no corpus."""
        return [cls.replace_matches(text, patterns_dict) for text in corpus]

    @staticmethod
    def remove_sites(raw_text: str) -> str:
        """Remove URLs do texto."""
        formatted_text = re.sub(
            r"(?:https?://)?(?:www\.)?[\w-]+\.[\w.-]+[^\s]*", "", raw_text
        )
        return " ".join(formatted_text.split())

    @classmethod
    def remove_punctuation(cls, raw_text: str) -> str:
        """Remove pontuações do texto."""
        doc = cls.nlp_object(raw_text)
        return " ".join([token.text for token in doc if not token.is_punct])

    @staticmethod
    def lower_text(raw_text: str) -> str:
        """Converte texto para minúsculas."""
        return raw_text.lower()

    @classmethod
    def remove_stopwords(cls, raw_text: str) -> str:
        """Remove stopwords do texto."""
        doc = cls.nlp_object(raw_text)
        return " ".join([token.text for token in doc if not token.is_stop])

    @classmethod
    def lemmatization(cls, raw_text: str) -> str:
        """Realiza lematização no texto."""
        doc = cls.nlp_object(raw_text)
        return " ".join([token.lemma_ for token in doc])

    @staticmethod
    def stemming(raw_text: str) -> str:
        """Realiza stemming no texto."""
        stemmer = nltk.stem.RSLPStemmer()
        return " ".join([stemmer.stem(token) for token in raw_text.split()])

    @staticmethod
    def remove_with_regex(raw_text: str, regex_patterns: List[str]) -> str:
        """Remove padrões regex do texto."""
        formatted_text = raw_text
        for pattern in regex_patterns:
            formatted_text = re.sub(pattern, "", formatted_text)
        return " ".join(formatted_text.split())

    @staticmethod
    def remove_with_prefixes(raw_text: str, prefixes: List[str]) -> str:
        """Remove palavras que começam com prefixos específicos."""
        words = []
        for word in raw_text.split():
            if word and word[0] not in prefixes:
                words.append(word)
        return " ".join(words)

    @classmethod
    def remove_emojis(cls, raw_text: str) -> str:
        """Remove emojis do texto."""
        doc = cls.nlp_object(raw_text)
        return " ".join([token.text for token in doc if not token._.is_emoji])

    @classmethod
    def replace_matches(
        cls, raw_text: str, patterns_dict: Dict[str, List[List[Dict[str, Any]]]]
    ) -> str:
        """Substitui padrões definidos no texto."""
        matcher = Matcher(cls.nlp_object.vocab)
        doc = cls.nlp_object(raw_text)
        for key, value in cls.__create_patterns_dict(patterns_dict).items():
            matcher.add(key, value)
        parsed_doc = doc.text
        for match_id, start, end in matcher(doc):
            string_id = cls.nlp_object.vocab.strings[match_id]
            span = doc[start:end]
            parsed_doc = parsed_doc.replace(span.text, string_id)
        return parsed_doc

    @classmethod
    def __create_patterns_dict(
        cls, patterns_data: Dict[str, List[List[Dict[str, Any]]]]
    ) -> Dict[str, List[List[Dict[str, Any]]]]:
        """Cria dicionário de padrões para Matcher."""
        patterns_dict = {}
        for pattern_name, emojis in patterns_data.items():
            pattern_list = [[{"ORTH": emoji}] for emoji in emojis]
            patterns_dict[pattern_name] = pattern_list
        return patterns_dict
