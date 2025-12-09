import json
import shutil
from pathlib import Path
from datetime import datetime
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.registry_dir / "manifest.json"
        logger.info(f" Model registry initialized: {registry_dir}")
    
    def compute_checksum(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def register_model(self, 
                      model_name: str,
                      model_version: str,
                      model_dir: str,
                      metadata: dict = None) -> dict:

        logger.info(f"\n{'='*80}")
        logger.info(f"REGISTERING MODEL: {model_name} v{model_version}")
        logger.info(f"{'='*80}")
        
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")
        
        version_dir = self.registry_dir / model_name / model_version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(" Copying model artifacts...")
        artifacts = {}
        
        for file_path in model_dir.glob("*"):
            if file_path.is_file():
                dest_path = version_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                
                checksum = self.compute_checksum(dest_path)
                artifacts[file_path.name] = {
                    'size_bytes': file_path.stat().st_size,
                    'checksum': checksum
                }
                logger.info(f"   {file_path.name} ({file_path.stat().st_size} bytes)")
        
        model_card = {
            'model_name': model_name,
            'model_version': model_version,
            'registered_at': datetime.now().isoformat(),
            'artifacts': artifacts,
            'metadata': metadata or {},
            'registry_path': str(version_dir)
        }
        
        card_path = version_dir / "model_card.json"
        with open(card_path, 'w') as f:
            json.dump(model_card, f, indent=2)
        
        logger.info(f"\n💾Model card saved: {card_path}")
        
        self._update_manifest(model_name, model_version, model_card)
        
        logger.info(f"\n{'='*80}")
        logger.info(f" MODEL REGISTERED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info(f"Registry path: {version_dir}")
        logger.info(f"Artifacts: {len(artifacts)}")
        
        return model_card
    
    def _update_manifest(self, model_name: str, version: str, model_card: dict):
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {'models': {}}
        
        if model_name not in manifest['models']:
            manifest['models'][model_name] = {'versions': []}
        
        version_entry = {
            'version': version,
            'registered_at': model_card['registered_at'],
            'artifact_count': len(model_card['artifacts'])
        }
        
        manifest['models'][model_name]['versions'] = [
            v for v in manifest['models'][model_name]['versions'] 
            if v['version'] != version
        ]
        
        manifest['models'][model_name]['versions'].append(version_entry)
        manifest['models'][model_name]['latest_version'] = version
        manifest['last_updated'] = datetime.now().isoformat()
        
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f" Manifest updated: {self.manifest_file}")
    
    def list_models(self):
        if not self.manifest_file.exists():
            logger.info("No models registered yet")
            return
        
        with open(self.manifest_file, 'r') as f:
            manifest = json.load(f)
        
        print("\n" + "="*80)
        print("📚 REGISTERED MODELS")
        print("="*80)
        
        for model_name, details in manifest['models'].items():
            print(f"\n {model_name}")
            print(f"   Latest version: {details['latest_version']}")
            print(f"   Total versions: {len(details['versions'])}")
            
            for version_info in details['versions']:
                print(f"   - v{version_info['version']} "
                      f"({version_info['artifact_count']} artifacts) "
                      f"registered {version_info['registered_at'][:10]}")
        
        print("\n" + "="*80)
    
    def get_model(self, model_name: str, version: str = None) -> dict:
        if version is None:
            with open(self.manifest_file, 'r') as f:
                manifest = json.load(f)
            version = manifest['models'][model_name]['latest_version']
        
        card_path = self.registry_dir / model_name / version / "model_card.json"
        
        with open(card_path, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" MODEL REGISTRATION SYSTEM")
    print("="*80)
    
    registry = ModelRegistry()
    
    print("\n📦 Registering embedding model...")
    embedding_card = registry.register_model(
        model_name="retrieval-embeddings",
        model_version="v1.0.0",
        model_dir="models/embeddings",
        metadata={
            'embedding_model': 'all-mpnet-base-v2',
            'embedding_dim': 384,
            'num_documents': 19,
            'framework': 'sentence-transformers',
            'purpose': 'GitLab onboarding assistant retrieval'
        }
    )
    
    print("\n Registering vector store metadata...")
    
    vs_metadata_dir = Path("models/vector_store_metadata")
    vs_metadata_dir.mkdir(parents=True, exist_ok=True)
    
    vs_info = {
        'vector_store_type': 'chromadb',
        'collection_name': 'gitlab_onboarding',
        'num_documents': 19,
        'persist_directory': 'models/vector_store',
        'distance_metric': 'cosine'
    }
    
    with open(vs_metadata_dir / "vector_store_info.json", 'w') as f:
        json.dump(vs_info, f, indent=2)
    
    vector_card = registry.register_model(
        model_name="vector-store",
        model_version="v1.0.0",
        model_dir="models/vector_store_metadata",
        metadata=vs_info
    )
    
    print("\n")
    registry.list_models()
    
    print("\n" + "="*80)
    print(" MODEL REGISTRATION COMPLETE!")
    print("="*80)
    print("\nNote: In production, these would be pushed to:")
    print("  - GCP Artifact Registry")
    print("  - GCS buckets with versioning")
    print("  - Vertex AI Model Registry")
    print("\nFor this project, we're using a local registry that")
    print("demonstrates the same versioning and tracking principles.")
    print("="*80)