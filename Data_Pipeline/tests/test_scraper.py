import unittest
from dags.scripts.scraper import parse_html_content

class TestScraper(unittest.TestCase):
    def test_parse_html_content(self):
        html = """
        <html>
            <head><title>Sample Page</title></head>
            <body>
                <h1>Welcome to GitLab</h1>
                <p>Transparency is one of our core values.</p>
                <p>We believe in collaboration.</p>
            </body>
        </html>
        """
        result = parse_html_content(html)
        
        self.assertIn("title", result)
        self.assertIn("paragraphs", result)
        self.assertTrue(isinstance(result["paragraphs"], list))
        self.assertGreater(len(result["paragraphs"]), 0)
        print("Scraper test passed!")

if __name__ == "__main__":
    unittest.main()
