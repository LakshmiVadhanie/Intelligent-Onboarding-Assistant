"""
YouTube Meeting Transcription Pipeline (Full-Length + Preprocessed)
===================================================================

Downloads full audio from up to N YouTube videos in a playlist
and transcribes them using Whisper + preprocessing.
Saves all transcripts into a combined JSON file ready for RAG.

Dependencies:
    - yt_dlp
    - openai-whisper
    - ffmpeg
"""

import os
import json
import yt_dlp
import whisper
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import sys

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("transcription")

# ---------------- Project Paths ---------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DATA_DIR = PROJECT_ROOT / "data"

# Import preprocessor (same as scraper)
try:
    from preprocess import default_preprocessor
except ImportError:
    sys.path.append(str(PROJECT_ROOT))
    from preprocess import default_preprocessor

# Configuration
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PL05JrBw4t0Kpap0GkV0SSuGnPhCM8jrAv"
OUTPUT_DIR = BASE_DATA_DIR / "meeting_transcripts"
AUDIO_DIR = BASE_DATA_DIR / "audio_files"
WHISPER_MODEL = "tiny"   # use small/base if you want better accuracy
MAX_VIDEOS = 5           # adjust as needed


# ---------------- Playlist & Audio Helpers ---------------- #
def get_playlist_videos(playlist_url: str) -> List[Dict[str, str]]:
    """Fetch all video IDs and titles from a YouTube playlist."""
    ydl_opts = {"quiet": True, "extract_flat": True, "force_generic_extractor": False}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        logger.info("Fetching playlist information...")
        playlist_info = ydl.extract_info(playlist_url, download=False)
        videos = [
            {"id": e["id"], "title": e.get("title", "Unknown Title")}
            for e in (playlist_info.get("entries") or [])
            if e
        ]
    logger.info(f"Found {len(videos)} videos. Using first {min(MAX_VIDEOS, len(videos))}.")
    return videos[:MAX_VIDEOS]


def download_audio_full(video_id: str, output_path_no_ext: str) -> bool:
    """Download full audio from a YouTube video."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "outtmpl": f"{output_path_no_ext}.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        logger.info(f"Downloaded full audio for video ID {video_id}")
        return True
    except Exception as e:
        logger.error(f"Download failed for video ID {video_id}: {e}")
        return False


def transcribe_audio(audio_path: str, model: whisper.Whisper) -> Tuple[str, List[Dict[str, Any]]]:
    """Transcribe audio file using Whisper."""
    try:
        logger.info(f"🎤 Transcribing → {audio_path}")
        result = model.transcribe(audio_path, fp16=False)
        return result["text"], result.get("segments", [])
    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {e}")
        return None, None


def clean_filename(filename: str, max_length: int = 100) -> str:
    safe_name = "".join(c for c in filename if c.isalnum() or c in (" ", "-", "_")).strip()
    return safe_name[:max_length]


# ---------------- Core Logic ---------------- #
def process_videos(videos: List[Dict[str, str]], output_dir: Path, audio_dir: Path, model: whisper.Whisper) -> None:
    """Download audio and transcribe videos into a combined JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    combined_path = output_dir / "all_transcripts.json"
    existing_items = []
    if combined_path.exists():
        with open(combined_path, "r", encoding="utf-8") as f:
            existing_items = json.load(f)

    existing_titles = {item.get("title") for item in existing_items}
    new_items = []

    logger.info(f"Starting processing of {len(videos)} videos...")

    for idx, video in enumerate(videos, 1):
        video_id, title = video["id"], video["title"]
        logger.info(f"[{idx}/{len(videos)}] {title}")

        if title in ("[Private video]", "[Deleted video]"):
            logger.warning(f"Skipping unavailable video: {title}")
            continue
        if title in existing_titles:
            logger.info(f"Already processed → skipping {title}")
            continue

        safe_title = clean_filename(title)
        audio_filename_no_ext = f"{idx:03d}_{safe_title}"
        audio_path_no_ext = str(audio_dir / audio_filename_no_ext)

        # Download full audio
        if not download_audio_full(video_id, audio_path_no_ext):
            logger.warning(f"Skipping {title} due to download failure.")
            continue

        audio_file = audio_path_no_ext + ".mp3"
        if not os.path.exists(audio_file):
            for ext in (".m4a", ".webm"):
                if os.path.exists(audio_path_no_ext + ext):
                    audio_file = audio_path_no_ext + ext
                    break
        if not os.path.exists(audio_file):
            logger.warning(f"Audio file missing for {title}")
            continue

        # Transcribe
        transcript_text, _ = transcribe_audio(audio_file, model)
        if not transcript_text:
            logger.warning(f"No transcript generated for {title}")
            continue

        # Preprocess transcript (same structure as scraper)
        pre = default_preprocessor.preprocess_for_scraper(title, [transcript_text])
        new_items.append({
            "title": pre["title"],
            "paragraph": " ".join(pre.get("paragraphs", []))
        })

        logger.info(f"✓ Transcript preprocessed and saved ({len(pre.get('paragraphs', []))} paragraphs)")

    all_items = existing_items + [it for it in new_items if it["title"] not in existing_titles]
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    logger.info(f"Combined JSON saved to {combined_path}")
    logger.info(f"Audio files stored in {audio_dir}")


# ---------------- CLI ---------------- #
def main():
    logger.info(f"Loading Whisper model: '{WHISPER_MODEL}' (fast inference)")
    model = whisper.load_model(WHISPER_MODEL)
    logger.info("✓ Whisper model loaded.")

    videos = get_playlist_videos(PLAYLIST_URL)
    logger.info(f"Processing up to {MAX_VIDEOS} videos (full length)...")

    process_videos(videos, OUTPUT_DIR, AUDIO_DIR, model)


if __name__ == "__main__":
    main()
