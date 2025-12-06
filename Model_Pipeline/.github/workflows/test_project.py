import pytest
from pathlib import Path

def test_project_structure():
    assert Path('src').exists(), "src directory should exist"
    assert Path('src/embeddings').exists(), "embeddings module should exist"
    assert Path('src/retrieval').exists(), "retrieval module should exist"
    assert Path('src/evaluation').exists(), "evaluation module should exist"
    assert Path('src/generation').exists(), "generation module should exist"
    assert Path('README.md').exists(), "README should exist"

def test_requirements_file():
    req_path = Path('requirements.txt')
    assert req_path.exists(), "requirements.txt should exist"
    
    with open(req_path) as f:
        content = f.read()
    assert len(content) > 0, "requirements.txt should not be empty"
    assert 'sentence-transformers' in content, "Should include sentence-transformers"

def test_model_directories():
    assert Path('models').exists(), "models directory should exist"
    assert Path('experiments').exists(), "experiments directory should exist"

def test_documentation():
    readme = Path('README.md')
    assert readme.exists(), "README.md should exist"
    
    with open(readme) as f:
        content = f.read()
    assert len(content) > 1000, "README should have substantial content"
    assert 'Installation' in content or 'Setup' in content, "README should have setup instructions"

def test_imports_available():
    try:
        import numpy
        assert True
    except ImportError:
        pytest.fail("NumPy should be importable")

if __name__ == "__main__":
    pytest.main([__file__, '-v'])