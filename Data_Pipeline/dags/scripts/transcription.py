"""
YouTube Meeting Transcription Pipeline
======================================

Downloads audio from a YouTube playlist and transcribes it using Whisper.
Automatically merges all transcripts into a combined JSON file.

Dependencies:
    - yt_dlp
    - openai-whisper
    - ffmpeg
"""

import os
import sys
import json
import yt_dlp
import whisper
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse

# ---------------- Logging Setup ---------------- #
# Ensure this script works both locally and inside Airflow Docker
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

try:
    from logging_utils import get_logger
except ImportError:
    from dags.scripts.logging_utils import get_logger  # fallback for Airflow package imports

logger = get_logger(__name__)

# ---------------- Paths (project-root aware) ---------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DATA_DIR = PROJECT_ROOT / "data"

# Configuration
PLAYLIST_URL = 'https://www.youtube.com/playlist?list=PL05JrBw4t0Kpap0GkV0SSuGnPhCM8jrAv'
OUTPUT_DIR = BASE_DATA_DIR / 'meeting_transcripts'
AUDIO_DIR = BASE_DATA_DIR / 'audio_files'
WHISPER_MODEL = 'base'


# ---------------- Playlist & Audio Helpers ---------------- #
def get_playlist_videos(playlist_url: str) -> List[Dict[str, str]]:
    """Fetch all video IDs and titles from a YouTube playlist."""
    ydl_opts = {'quiet': True, 'extract_flat': True, 'force_generic_extractor': False}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        logger.info("Fetching playlist information...")
        playlist_info = ydl.extract_info(playlist_url, download=False)

        videos: List[Dict[str, str]] = []
        for entry in (playlist_info.get('entries') or []):
            if entry:
                videos.append({'id': entry['id'], 'title': entry.get('title', 'Unknown Title')})
        logger.info(f"Found {len(videos)} videos in playlist.")
        return videos


def download_audio(video_id: str, output_path_no_ext: str) -> bool:
    """Download audio from a YouTube video to MP3 (or best available)."""
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': output_path_no_ext,
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        logger.info(f"Downloaded audio for video ID {video_id}")
        return True
    except Exception as e:
        logger.exception(f"Download failed for video ID {video_id}: {e}")
        return False


def transcribe_audio(audio_path: str, model: whisper.Whisper) -> Tuple[str, List[Dict[str, Any]]]:
    """Transcribe audio file using local Whisper."""
    try:
        logger.info(f"🎤 Transcribing audio with Whisper ({WHISPER_MODEL}) → {audio_path}")
        result = model.transcribe(audio_path, fp16=False)
        logger.info(f"Transcription completed for {os.path.basename(audio_path)}")
        return result['text'], result.get('segments', [])
    except Exception as e:
        logger.exception(f"Transcription failed for {audio_path}: {e}")
        return None, None


def clean_filename(filename: str, max_length: int = 100) -> str:
    """Create a safe filename."""
    safe_name = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).strip()
    return safe_name[:max_length]


# ---------------- Core Processing Logic ---------------- #
def process_videos(videos: List[Dict[str, str]],
                   output_dir: Path,
                   audio_dir: Path,
                   whisper_model: whisper.Whisper) -> None:
    """Download audio and transcribe all videos into a combined JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    successful = failed = skipped = 0
    failed_videos = []
    new_items = []

    logger.info(f"Processing {len(videos)} videos...")

    combined_path = output_dir / 'all_transcripts.json'
    if combined_path.exists():
        with open(combined_path, 'r', encoding='utf-8') as f:
            existing_items = json.load(f)
    else:
        existing_items = []

    existing_titles = {item.get('title') for item in existing_items}

    for idx, video in enumerate(videos, 1):
        video_id = video['id']
        title = video['title']
        logger.info(f"[{idx}/{len(videos)}] {title}")

        # Skip unavailable or duplicate
        if title in ('[Private video]', '[Deleted video]'):
            logger.warning(f"Skipping unavailable video: {title}")
            skipped += 1
            continue
        if title in existing_titles:
            logger.info(f"Already processed → skipping {title}")
            successful += 1
            continue

        safe_title = clean_filename(title)
        audio_filename_no_ext = f"{idx:03d}_{safe_title}"
        audio_path_no_ext = str(audio_dir / audio_filename_no_ext)

        # Download
        logger.info("⬇ Downloading audio...")
        if not download_audio(video_id, audio_path_no_ext):
            failed += 1
            failed_videos.append({'title': title, 'url': f'https://www.youtube.com/watch?v={video_id}'})
            continue

        # Find file
        audio_file = None
        for ext in ('.mp3', '.m4a', '.webm'):
            potential = audio_path_no_ext + ext
            if os.path.exists(potential):
                audio_file = potential
                break
        if not audio_file:
            logger.error(f"Audio file not found for {title}")
            failed += 1
            failed_videos.append({'title': title, 'url': f'https://www.youtube.com/watch?v={video_id}'})
            continue

        # Transcribe
        transcript_text, _ = transcribe_audio(audio_file, whisper_model)
        if transcript_text:
            new_items.append({
                'title': title,
                'video_id': video_id,
                'url': f'https://www.youtube.com/watch?v={video_id}',
                'transcript': transcript_text
            })
            logger.info(f"✓ Transcript captured ({len(transcript_text)} characters)")
            successful += 1
        else:
            failed += 1
            failed_videos.append({'title': title, 'url': f'https://www.youtube.com/watch?v={video_id}'})

    # Merge results
    all_items = existing_items + [it for it in new_items if it['title'] not in existing_titles]
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)
    logger.info(f"Combined JSON saved to {combined_path}")

    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f" Successful: {successful}/{len(videos)}")
    logger.info(f" Failed: {failed}/{len(videos)}")
    logger.info(f" Skipped (unavailable or duplicate): {skipped}/{len(videos)}")

    if failed_videos:
        preview = ', '.join(v['title'] for v in failed_videos[:5])
        logger.warning(f"Failed videos: {preview}{' ...' if len(failed_videos) > 5 else ''}")

    logger.info(f"Audio files saved to: {audio_dir}")


# ---------------- CLI Entry ---------------- #
def main():
    parser = argparse.ArgumentParser(description="Transcribe a YouTube playlist with Whisper.")
    parser.add_argument("--limit", type=int, default=3,
                        help="Max number of videos to process (default: 3)")
    args = parser.parse_args()

    logger.info("Loading Whisper model...")
    logger.info(f"Using '{WHISPER_MODEL}' model. This may take a moment on first run...")

    try:
        model = whisper.load_model(WHISPER_MODEL)
        logger.info("✓ Whisper model loaded successfully.")

        videos = get_playlist_videos(PLAYLIST_URL)
        if args.limit and args.limit > 0:
            logger.info(f"Limiting to first {args.limit} videos.")
            videos = videos[:args.limit]

        process_videos(videos, OUTPUT_DIR, AUDIO_DIR, model)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        logger.error("Make sure ffmpeg and whisper dependencies are installed properly.")


if __name__ == "__main__":
    main()
