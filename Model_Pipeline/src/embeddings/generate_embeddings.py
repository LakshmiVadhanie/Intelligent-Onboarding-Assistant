from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
 
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded! Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:

        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  
        )
        
        logger.info(f" Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def process_chunks(self, chunks: List[Dict], text_field: str = "paragraph") -> Tuple[List[str], np.ndarray, List[Dict]]:

        logger.info(f"Processing {len(chunks)} chunks...")
        
        texts = []
        metadata = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.get(text_field, chunk.get('text', chunk.get('content', '')))
            
            if not text or not text.strip():
                logger.warning(f"Empty text in chunk {i}, skipping...")
                continue
            
            texts.append(text)
            
            meta = {
                'chunk_id': chunk.get('chunk_id', f'chunk_{i}'),
                'title': chunk.get('title', ''),
                'source': chunk.get('source', 'unknown'),
                'source_type': chunk.get('source_type', 'unknown'),
                'url': chunk.get('url', ''),
                'original_index': i
            }
            metadata.append(meta)
        
        logger.info(f"Extracted {len(texts)} valid texts")
        
        embeddings = self.generate_embeddings(texts)
        
        return texts, embeddings, metadata
    
    def save_embeddings(self, 
                       texts: List[str],
                       embeddings: np.ndarray, 
                       metadata: List[Dict], 
                       output_dir: str = "models/embeddings"):
  
        os.makedirs(output_dir, exist_ok=True)
        
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        logger.info(f" Saved embeddings to: {embeddings_path}")
        
        texts_path = os.path.join(output_dir, "texts.json")
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        logger.info(f" Saved texts to: {texts_path}")
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f" Saved metadata to: {metadata_path}")
        
        model_info = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'num_embeddings': len(embeddings),
            'num_texts': len(texts)
        }
        info_path = os.path.join(output_dir, "model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        logger.info(f" Saved model info to: {info_path}")
        
        logger.info(f" All files saved successfully to: {output_dir}")
    
    def load_embeddings(self, input_dir: str = "models/embeddings") -> Tuple[List[str], np.ndarray, List[Dict]]:

        embeddings_path = os.path.join(input_dir, "embeddings.npy")
        texts_path = os.path.join(input_dir, "texts.json")
        metadata_path = os.path.join(input_dir, "metadata.json")
        
        embeddings = np.load(embeddings_path)
        
        with open(texts_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f" Loaded {len(embeddings)} embeddings from: {input_dir}")
        return texts, embeddings, metadata


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.load_local_data import LocalDataLoader
    
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data...")
    logger.info("=" * 60)
    data_loader = LocalDataLoader(data_dir="data")
    chunks = data_loader.load_all_data()
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Generating embeddings...")
    logger.info("=" * 60)
    embedding_gen = EmbeddingGenerator(model_name="all-mpnet-base-v2")
    texts, embeddings, metadata = embedding_gen.process_chunks(chunks)
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Saving embeddings...")
    logger.info("=" * 60)
    embedding_gen.save_embeddings(texts, embeddings, metadata)
    
    print("\n" + "=" * 60)
    print(" EMBEDDING GENERATION COMPLETE!")
    print("=" * 60)
    print(f" Total chunks processed: {len(texts)}")
    print(f" Embedding dimensions: {embeddings.shape}")
    print(f" Files saved to: models/embeddings/")
    print("=" * 60)