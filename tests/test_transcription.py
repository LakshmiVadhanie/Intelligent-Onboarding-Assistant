import unittest
from dags.scripts.transcription import clean_filename, transcribe_audio

class DummyModel:
    def transcribe(self, audio_path, fp16=False):
        return {"text": "This is a dummy transcription.", "segments": []}

class TestTranscription(unittest.TestCase):
    def test_clean_filename(self):
        name = clean_filename("GitLab: Values & Culture!!")
        self.assertTrue(name)
        self.assertNotIn(":", name)
        print("Filename cleaning test passed!")

    def test_transcribe_audio(self):
        model = DummyModel()
        text, segments = transcribe_audio("fake_path.mp3", model)
        # even if path is fake, DummyModel works
        self.assertIsInstance(text, str)
        print("Transcription mock test passed!")

if __name__ == "__main__":
    unittest.main()
