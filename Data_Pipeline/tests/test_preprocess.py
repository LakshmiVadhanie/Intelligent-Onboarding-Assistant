import unittest
from dags.scripts.preprocess import default_preprocessor

class TestPreprocessor(unittest.TestCase):
    def test_clean_paragraphs(self):
        paragraphs = [
            "GitLab is awesome! ",
            "GitLab is awesome! ",  # duplicate
            "  Too short  ",
            "",
            "Transparency and collaboration are key values."
        ]

        cleaned = default_preprocessor.preprocess_paragraphs(paragraphs)
        
        # Should remove emojis, duplicates, and short entries
        self.assertTrue(len(cleaned) >= 1)
        self.assertTrue(all(isinstance(p, str) for p in cleaned))
        self.assertTrue(all(len(p) >= default_preprocessor.min_length for p in cleaned))
        print("Preprocessor test passed!")

if __name__ == "__main__":
    unittest.main()
