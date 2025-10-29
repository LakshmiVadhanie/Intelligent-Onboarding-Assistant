#!/usr/bin/env python3
"""
Test the complete pipeline locally (without Airflow)
This simulates what the Airflow DAG would do
"""

import sys
from pathlib import Path
import time
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path.cwd() / "data-pipeline"))

print("=" * 60)
print("INTELLIGENT ONBOARDING ASSISTANT - PIPELINE TEST")
print("=" * 60)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 60)

# Track results
results = {}

# Step 1: Environment Check
print("\n📋 STEP 1: Checking Environment...")
print("-" * 40)
try:
    required_dirs = [
        'data-pipeline/data/raw',
        'data-pipeline/data/processed/cleaned',
        'data-pipeline/data/processed/chunked',
        'data-pipeline/data/curated',
        'data-pipeline/logs/pipeline'
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {dir_path}")
    
    results['environment'] = 'ready'
    print("✓ Environment check complete!")
except Exception as e:
    print(f"❌ Environment check failed: {e}")
    results['environment'] = 'failed'

# Step 2: Data Ingestion Status
print("\n📥 STEP 2: Checking Data Ingestion...")
print("-" * 40)

# Check handbook data
handbook_dir = Path("data-pipeline/data/raw/handbook")
handbook_files = list(handbook_dir.glob("handbook_*.json")) if handbook_dir.exists() else []
print(f"✅ GitLab Handbook: {len(handbook_files)} documents")
results['handbook'] = len(handbook_files)

# Check transcripts
transcript_dir = Path("data-pipeline/data/raw/transcripts")
transcript_files = list(transcript_dir.glob("transcript_*.json")) if transcript_dir.exists() else []
print(f"✅ YouTube Transcripts: {len(transcript_files)} videos")
results['transcripts'] = len(transcript_files)

# Check blogs
blog_dir = Path("data-pipeline/data/raw/blogs")
blog_files = list(blog_dir.glob("blog_*.json")) if blog_dir.exists() else []
print(f"✅ Blog Posts: {len(blog_files)} articles")
results['blogs'] = len(blog_files)

total_raw = len(handbook_files) + len(transcript_files) + len(blog_files)
print(f"\n📊 Total raw documents: {total_raw}")

# Step 3: Run Preprocessing (if needed)
print("\n🔧 STEP 3: Preprocessing Pipeline...")
print("-" * 40)

# Check if cleaning is needed
cleaned_dir = Path("data-pipeline/data/processed/cleaned")
cleaned_files = list(cleaned_dir.glob("cleaned_*.json")) if cleaned_dir.exists() else []

if len(cleaned_files) == 0 and len(handbook_files) > 0:
    print("Running text cleaning...")
    try:
        from scripts.preprocessing.text_cleaner import TextCleaner
        cleaner = TextCleaner()
        processed_docs = cleaner.process_all_handbook_documents()
        print(f"✅ Cleaned {len(processed_docs)} documents")
        results['cleaned'] = len(processed_docs)
    except Exception as e:
        print(f"⚠️ Cleaning failed: {e}")
        results['cleaned'] = 0
else:
    print(f"✅ Already cleaned: {len(cleaned_files)} documents")
    results['cleaned'] = len(cleaned_files)

# Check if chunking is needed
chunked_dir = Path("data-pipeline/data/processed/chunked")
chunked_files = list(chunked_dir.glob("chunks_*.json")) if chunked_dir.exists() else []

if len(chunked_files) == 0 and len(cleaned_files) > 0:
    print("Running document chunking...")
    try:
        from scripts.preprocessing.chunking_strategy import TextChunker
        chunker = TextChunker()
        chunks = chunker.process_all_documents()
        print(f"✅ Created {len(chunks)} chunks")
        results['chunks'] = len(chunks)
    except Exception as e:
        print(f"⚠️ Chunking failed: {e}")
        results['chunks'] = 0
else:
    print(f"✅ Already chunked: {len(chunked_files)} files")
    # Count actual chunks
    total_chunks = 0
    for file in chunked_files[:3]:  # Sample a few files
        import json
        with open(file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                total_chunks += len(data)
    results['chunks'] = total_chunks

# Step 4: Validation
print("\n✅ STEP 4: Data Validation...")
print("-" * 40)

# Basic validation
validation_results = {
    'has_raw_data': total_raw > 0,
    'has_cleaned_data': len(cleaned_files) > 0,
    'has_chunks': len(chunked_files) > 0
}

for check, status in validation_results.items():
    symbol = "✅" if status else "❌"
    print(f"{symbol} {check.replace('_', ' ').title()}: {status}")

results['validation'] = all(validation_results.values())

# Step 5: Generate Summary
print("\n📊 STEP 5: Pipeline Summary...")
print("-" * 40)

summary = {
    'timestamp': datetime.now().isoformat(),
    'raw_data': {
        'handbook': results.get('handbook', 0),
        'transcripts': results.get('transcripts', 0),
        'blogs': results.get('blogs', 0),
        'total': total_raw
    },
    'processed_data': {
        'cleaned': results.get('cleaned', 0),
        'chunks': results.get('chunks', 0)
    },
    'validation': results.get('validation', False)
}

# Save summary
import json
summary_path = Path("data-pipeline/data/pipeline_summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to: {summary_path}")
print("\nPipeline Statistics:")
print(json.dumps(summary, indent=2))

# Final Status
print("\n" + "=" * 60)
if summary['validation']:
    print("✅ PIPELINE TEST SUCCESSFUL!")
    print("\nYour data pipeline is working correctly.")
    print("\nNext steps:")
    print("1. Set up Airflow to automate this pipeline")
    print("2. Add more validation and testing")
    print("3. Implement DVC for version control")
else:
    print("⚠️ PIPELINE TEST INCOMPLETE")
    print("\nSome components are missing data.")
    print("Check the summary above for details.")

print("=" * 60)
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)