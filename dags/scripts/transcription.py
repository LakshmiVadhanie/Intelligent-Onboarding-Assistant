import os
import json
import yt_dlp
import whisper
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse

# --- Paths (project-root aware) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DATA_DIR = PROJECT_ROOT / "data"

# Configuration
PLAYLIST_URL = 'https://www.youtube.com/playlist?list=PL05JrBw4t0Kpap0GkV0SSuGnPhCM8jrAv'
OUTPUT_DIR = BASE_DATA_DIR / 'meeting_transcripts'
AUDIO_DIR = BASE_DATA_DIR / 'audio_files'
WHISPER_MODEL = 'base'


def get_playlist_videos(playlist_url: str) -> List[Dict[str, str]]:
    """Fetch all video IDs and titles from a YouTube playlist."""
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print("Fetching playlist information...")
        playlist_info = ydl.extract_info(playlist_url, download=False)

        videos: List[Dict[str, str]] = []
        for entry in (playlist_info.get('entries') or []):
            if entry:
                videos.append({
                    'id': entry['id'],
                    'title': entry.get('title', 'Unknown Title')
                })
        return videos


def download_audio(video_id: str, output_path_no_ext: str) -> bool:
    """Download audio from a YouTube video to MP3 (or best available)."""
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path_no_ext,
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {str(e)}")
        return False


def transcribe_audio(audio_path: str, model: whisper.Whisper) -> Tuple[str, List[Dict[str, Any]]]:
    """Transcribe audio file using local Whisper."""
    try:
        print(f"  🎤 Transcribing with Whisper ({WHISPER_MODEL} model)...")
        result = model.transcribe(audio_path, fp16=False)
        return result['text'], result.get('segments', [])
    except Exception as e:
        print(f"  ✗ Transcription failed: {str(e)}")
        return None, None


def clean_filename(filename: str, max_length: int = 100) -> str:
    """Create a safe filename."""
    safe_name = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).strip()
    return safe_name[:max_length]


def process_videos(videos: List[Dict[str, str]],
                   output_dir: Path,
                   audio_dir: Path,
                   whisper_model: whisper.Whisper) -> None:
    """Download audio and transcribe all videos into a combined JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    successful = 0
    failed = 0
    skipped = 0
    failed_videos = []
    new_items = []

    print(f"\nProcessing {len(videos)} videos...\n")

    # Load existing combined JSON if present
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

        print(f"\n[{idx}/{len(videos)}] {title}")

        # Skip private/unavailable
        if title in ('[Private video]', '[Deleted video]'):
            print("  ⊘ Skipping unavailable video")
            skipped += 1
            continue

        # Skip if already present in combined JSON
        if title in existing_titles:
            print("  ✓ Already in all_transcripts.json, skipping…")
            successful += 1
            continue

        # Prepare output names
        safe_title = clean_filename(title)
        audio_filename_no_ext = f"{idx:03d}_{safe_title}"
        audio_path_no_ext = str(audio_dir / audio_filename_no_ext)

        # Download audio
        print("  ⬇ Downloading audio…")
        if not download_audio(video_id, audio_path_no_ext):
            failed += 1
            failed_videos.append({'title': title, 'url': f'https://www.youtube.com/watch?v={video_id}'})
            continue

        # Find actual downloaded file (yt-dlp adds extension)
        audio_file = None
        for ext in ('.mp3', '.m4a', '.webm'):
            potential_file = audio_path_no_ext + ext
            if os.path.exists(potential_file):
                audio_file = potential_file
                break

        if not audio_file:
            print("  ✗ Audio file not found")
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
            print(f"  ✓ Transcript captured ({len(transcript_text)} characters)")
            successful += 1
        else:
            failed += 1
            failed_videos.append({'title': title, 'url': f'https://www.youtube.com/watch?v={video_id}'})

    # Merge & write combined JSON (unchanged behavior)
    all_items = existing_items + [it for it in new_items if it['title'] not in existing_titles]
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successfully processed: {successful}/{len(videos)}")
    print(f"Failed: {failed}/{len(videos)}")
    print(f"Skipped (unavailable or already present): {skipped}/{len(videos)}")

    if failed_videos:
        print("\nFailed videos (first few):")
        for video in failed_videos[:5]:
            print(f"  - {video['title']}")
        if len(failed_videos) > 5:
            print(f"  ... and {len(failed_videos) - 5} more")

    print(f"\nCombined JSON saved to: {combined_path}")
    print(f"Audio files saved to: {audio_dir}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe a YouTube playlist with Whisper.")
    parser.add_argument("--limit", type=int, default=3,
                        help="Max number of videos to process (default: 3)")
    args = parser.parse_args()

    print("Loading Whisper model...")
    print(f"Using '{WHISPER_MODEL}' model. This may take a moment on first run...")

    try:
        model = whisper.load_model(WHISPER_MODEL)
        print("✓ Model loaded successfully!\n")

        videos = get_playlist_videos(PLAYLIST_URL)
        print(f"Found {len(videos)} videos in playlist")

        # Non-interactive: always cap by --limit (default 3)
        if args.limit is not None and args.limit > 0:
            print(f"\nLimiting to first {args.limit} videos for this run.")
            videos = videos[:args.limit]

        process_videos(videos, OUTPUT_DIR, AUDIO_DIR, model)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nMake sure you have installed:")
        print("pip install openai-whisper yt-dlp")
        print("And install ffmpeg:\n  Mac: brew install ffmpeg\n  Ubuntu: sudo apt install ffmpeg")


if __name__ == "__main__":
    main()


# import os
# import json
# import yt_dlp
# import whisper
# from pathlib import Path
# import sys

# # -----------------------------
# # Path Configuration
# # -----------------------------
# PROJECT_ROOT = Path(__file__).resolve().parents[2]   # Goes two levels up from dags/scripts/
# BASE_DATA_DIR = PROJECT_ROOT / "data"
# OUTPUT_DIR = BASE_DATA_DIR / "meeting_transcripts"
# AUDIO_DIR = BASE_DATA_DIR / "audio_files"
# WHISPER_MODEL = 'base'

# # Import preprocessor
# sys.path.append(str(Path(__file__).resolve().parents[0]))
# from preprocess import default_preprocessor

# # -----------------------------
# # Playlist Configuration
# # -----------------------------
# PLAYLIST_URL = 'https://www.youtube.com/playlist?list=PL05JrBw4t0Kpap0GkV0SSuGnPhCM8jrAv'

# # -----------------------------
# # Utility Functions
# # -----------------------------
# def get_playlist_videos(playlist_url):
#     """Fetch all video IDs and titles from a YouTube playlist."""
#     ydl_opts = {
#         'quiet': True,
#         'extract_flat': True,
#         'force_generic_extractor': False,
#     }
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         print("Fetching playlist information...")
#         playlist_info = ydl.extract_info(playlist_url, download=False)
#         videos = []
#         for entry in playlist_info['entries']:
#             if entry:
#                 videos.append({
#                     'id': entry['id'],
#                     'title': entry.get('title', 'Unknown Title')
#                 })
#         return videos


# def download_audio(video_id, output_path):
#     """Download audio from a YouTube video."""
#     video_url = f'https://www.youtube.com/watch?v={video_id}'
#     ydl_opts = {
#         'format': 'bestaudio/best',
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'mp3',
#             'preferredquality': '192',
#         }],
#         'outtmpl': output_path,
#         'quiet': True,
#         'no_warnings': True,
#     }
#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([video_url])
#         return True
#     except Exception as e:
#         print(f"  ✗ Download failed: {str(e)}")
#         return False


# def transcribe_audio(audio_path, model):
#     """Transcribe audio file using local Whisper."""
#     try:
#         print(f"  🎤 Transcribing with Whisper ({WHISPER_MODEL} model)...")
#         result = model.transcribe(audio_path, fp16=False)
#         return result['text'], result.get('segments', [])
#     except Exception as e:
#         print(f"  ✗ Transcription failed: {str(e)}")
#         return None, None


# def clean_filename(filename, max_length=100):
#     """Create a safe filename."""
#     safe_name = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).strip()
#     return safe_name[:max_length]


# # -----------------------------
# # Core Processing Logic
# # -----------------------------
# def process_videos(videos, output_dir, audio_dir, whisper_model):
#     """Download audio and transcribe all videos."""
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(audio_dir, exist_ok=True)

#     successful = 0
#     failed = 0
#     skipped = 0
#     failed_videos = []
#     items = []

#     print(f"\nProcessing {len(videos)} videos...\n")

#     for idx, video in enumerate(videos, 1):
#         video_id = video['id']
#         title = video['title']

#         print(f"\n[{idx}/{len(videos)}] {title}")

#         if title == '[Private video]':
#             print("  ⊘ Skipping private video")
#             skipped += 1
#             continue

#         safe_title = clean_filename(title)
#         audio_filename = f"{idx:03d}_{safe_title}"
#         audio_path = os.path.join(audio_dir, audio_filename)

#         # Skip if already in JSON (if rerun)
#         combined_path = os.path.join(output_dir, 'all_transcripts.json')
#         if os.path.exists(combined_path):
#             with open(combined_path, 'r', encoding='utf-8') as f:
#                 existing_data = json.load(f)
#                 if any(item['title'] == title for item in existing_data):
#                     print("  ✓ Already transcribed, skipping...")
#                     successful += 1
#                     continue

#         # Download audio
#         print("  ⬇ Downloading audio...")
#         if not download_audio(video_id, audio_path):
#             failed += 1
#             failed_videos.append({'title': title, 'url': f'https://www.youtube.com/watch?v={video_id}'})
#             continue

#         # Find actual downloaded file (yt-dlp adds extension)
#         audio_file = None
#         for ext in ['.mp3', '.m4a', '.webm']:
#             potential_file = audio_path + ext
#             if os.path.exists(potential_file):
#                 audio_file = potential_file
#                 break

#         if not audio_file:
#             print("  ✗ Audio file not found")
#             failed += 1
#             failed_videos.append({'title': title, 'url': f'https://www.youtube.com/watch?v={video_id}'})
#             continue

#         # Transcribe
#         transcript_text, _ = transcribe_audio(audio_file, whisper_model)

#         if transcript_text:
#             # 🧹 Clean transcript using preprocess.py
#             cleaned_text = " ".join(
#                 default_preprocessor.preprocess_paragraphs([transcript_text])
#             )

#             items.append({
#                 'title': title,
#                 'video_id': video_id,
#                 'url': f'https://www.youtube.com/watch?v={video_id}',
#                 'transcript': cleaned_text
#             })

#             print(f"  ✓ Transcript captured ({len(cleaned_text)} characters)")
#             successful += 1
#         else:
#             failed += 1
#             failed_videos.append({'title': title, 'url': f'https://www.youtube.com/watch?v={video_id}'})

#     # Combine with existing JSON
#     combined_path = os.path.join(output_dir, 'all_transcripts.json')
#     if os.path.exists(combined_path):
#         with open(combined_path, 'r', encoding='utf-8') as f:
#             existing_items = json.load(f)
#     else:
#         existing_items = []

#     existing_titles = {item['title'] for item in existing_items}
#     all_items = existing_items + [item for item in items if item['title'] not in existing_titles]

#     with open(combined_path, 'w', encoding='utf-8') as f:
#         json.dump(all_items, f, indent=2, ensure_ascii=False)

#     # Summary
#     print("\n" + "=" * 80)
#     print("SUMMARY")
#     print("=" * 80)
#     print(f"Successfully processed: {successful}/{len(videos)}")
#     print(f"Failed: {failed}/{len(videos)}")
#     print(f"Skipped (private): {skipped}/{len(videos)}")

#     if failed_videos:
#         print(f"\nFailed videos:")
#         for video in failed_videos[:5]:
#             print(f"  - {video['title']}")
#         if len(failed_videos) > 5:
#             print(f"  ... and {len(failed_videos) - 5} more")

#     print(f"\nCombined JSON saved to: {combined_path}")
#     print(f"Audio files saved to '{audio_dir}/' directory")


# # -----------------------------
# # Entry Point
# # -----------------------------
# def main():
#     print("Loading Whisper model...")
#     print(f"Using '{WHISPER_MODEL}' model. This may take a moment on first run...")

#     try:
#         model = whisper.load_model(WHISPER_MODEL)
#         print("✓ Model loaded successfully!\n")

#         videos = get_playlist_videos(PLAYLIST_URL)
#         print(f"Found {len(videos)} videos in playlist")

#         response = input(f"\nProcess all {len(videos)} videos? (y/n): ")
#         if response.lower() != 'y':
#             try:
#                 num = int(input("How many videos do you want to process? "))
#                 videos = videos[:num]
#             except:
#                 print("Invalid input, processing first 5 videos")
#                 videos = videos[:5]

#         process_videos(videos, OUTPUT_DIR, AUDIO_DIR, model)

#     except Exception as e:
#         print(f"\nError: {str(e)}")
#         print("\nMake sure you have installed:")
#         print("pip install openai-whisper yt-dlp")
#         print("And install ffmpeg:\n  Mac: brew install ffmpeg\n  Ubuntu: sudo apt install ffmpeg")


# if __name__ == "__main__":
#     main()

