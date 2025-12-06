from google.cloud import storage
import json
import logging
from typing import List, Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCSDataLoader:
    def __init__(self, bucket_name: str, project_id: str = None):
 
        self.bucket_name = bucket_name
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
        logger.info(f"Connected to GCS bucket: {bucket_name}")
    
    def load_chunked_data(self, prefix: str = "data/processed/") -> List[Dict]:

        chunks = []
        
        blobs = self.bucket.list_blobs(prefix=prefix)
        
        for blob in blobs:
            if blob.name.endswith('.json'):
                logger.info(f"Loading: {blob.name}")
                
                content = blob.download_as_text()
                data = json.loads(content)
                
                if isinstance(data, list):
                    chunks.extend(data)
                else:
                    chunks.append(data)
        
        logger.info(f"Loaded {len(chunks)} chunks from GCS")
        return chunks
    
    def load_handbook_chunks(self) -> List[Dict]:
        return self.load_chunked_data(prefix="data/processed/handbook/")
    
    def load_transcript_chunks(self) -> List[Dict]:
        return self.load_chunked_data(prefix="data/processed/transcripts/")
    
    def load_all_chunks(self) -> List[Dict]:
        logger.info("Loading all chunks from GCS...")
        
        handbook_chunks = self.load_handbook_chunks()
        transcript_chunks = self.load_transcript_chunks()
        
        all_chunks = handbook_chunks + transcript_chunks
        logger.info(f"Total chunks loaded: {len(all_chunks)}")
        
        return all_chunks


if __name__ == "__main__":
    BUCKET_NAME = "your-bucket-name" 
    PROJECT_ID = "your-project-id"    
    
    loader = GCSDataLoader(
        bucket_name=BUCKET_NAME,
        project_id=PROJECT_ID
    )
    
    chunks = loader.load_all_chunks()
    
    print(f"\n Total chunks loaded: {len(chunks)}")
    
    if chunks:
        print(f"\n Sample chunk:")
        print(json.dumps(chunks[0], indent=2))