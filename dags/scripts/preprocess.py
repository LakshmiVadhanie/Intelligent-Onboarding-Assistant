import re
import html
import unicodedata
from typing import List, Dict, Any


class ParagraphPreprocessor:
    def __init__(
        self,
        min_length: int = 30,
        remove_emojis: bool = True,
        dedupe: bool = True,
        collapse_whitespace: bool = True,
        lowercase: bool = False,
    ) -> None:
        self.min_length = min_length
        self.remove_emojis = remove_emojis
        self.dedupe = dedupe
        self.collapse_whitespace = collapse_whitespace
        self.lowercase = lowercase

        self._ws_re = re.compile(r"\s+")
        self._emoji_re = re.compile(r"[\u2600-\u27BF\U0001F300-\U0001FAFF]")

    def _clean_text(self, text: str) -> str:
        s = html.unescape(text)
        s = unicodedata.normalize("NFKC", s)
        if self.lowercase:
            s = s.lower()
        if self.remove_emojis:
            s = self._emoji_re.sub("", s)
        if self.collapse_whitespace:
            s = self._ws_re.sub(" ", s)
        return s.strip()

    def preprocess_title(self, title: str) -> str:
        return self._clean_text(title) if title else ""

    def preprocess_paragraphs(self, paragraphs: List[str]) -> List[str]:
        if not paragraphs:
            return []

        seen = set()
        cleaned: List[str] = []
        for p in paragraphs:
            if not p:
                continue
            cp = self._clean_text(p)
            if not cp or len(cp) < self.min_length:
                continue
            if self.dedupe:
                if cp in seen:
                    continue
                seen.add(cp)
            cleaned.append(cp)
        return cleaned

    def preprocess_for_scraper(self, title: str, paragraphs: List[str]) -> Dict[str, Any]:
        return {
            "title": self.preprocess_title(title),
            "paragraphs": self.preprocess_paragraphs(paragraphs),
        }


default_preprocessor = ParagraphPreprocessor()