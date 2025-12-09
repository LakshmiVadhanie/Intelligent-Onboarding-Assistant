import os
import json
import yt_dlp
import whisper
from pathlib import Path

# Configuration
PLAYLIST_URL = 'https://www.youtube.com/playlist?list=PL05JrBw4t0Kpap0GkV0SSuGnPhCM8jrAv'
OUTPUT_DIR = 'meeting_transcripts'
AUDIO_DIR = 'audio_files'
WHISPER_MODEL = 'base'  # Options: tiny, base, small, medium, large
                        # tiny: fastest, least accurate
                        # base: good balance (recommended to start)
                        # large: most accurate, slowest

def get_playlist_videos(playlist_url):
    """Fetch all video IDs and titles from a YouTube playlist."""
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print("Fetching playlist information...")
        playlist_info = ydl.extract_info(playlist_url, download=False)
        
        videos = []
        for entry in playlist_info['entries']:
            if entry:
                videos.append({
                    'id': entry['id'],
                    'title': entry.get('title', 'Unknown Title')
                })
        
        return videos

def download_audio(video_id, output_path):
    """Download audio from a YouTube video."""
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
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

def transcribe_audio(audio_path, model):
    """Transcribe audio file using local Whisper."""
    try:
        print(f"  🎤 Transcribing with Whisper ({WHISPER_MODEL} model)...")
        result = model.transcribe(audio_path, fp16=False)
        return result['text'], result.get('segments', [])
    except Exception as e:
        print(f"  ✗ Transcription failed: {str(e)}")
        return None, None

def clean_filename(filename, max_length=100):
    """Create a safe filename."""
    safe_name = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).strip()
    return safe_name[:max_length]

def process_videos(videos, output_dir, audio_dir, whisper_model):
    """Download audio and transcribe all videos."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    skipped = 0
    failed_videos = []
    
    print(f"\nProcessing {len(videos)} videos...\n")
    
    for idx, video in enumerate(videos, 1):
        video_id = video['id']
        title = video['title']
        
        print(f"\n[{idx}/{len(videos)}] {title}")
        
        # Skip private videos
        if title == '[Private video]':
            print("  ⊘ Skipping private video")
            skipped += 1
            continue
        
        # Prepare filenames
        safe_title = clean_filename(title)
        txt_filename = f"{idx:03d}_{safe_title}.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        # Skip if already transcribed
        if os.path.exists(txt_path):
            print("  ✓ Already transcribed, skipping...")
            successful += 1
            continue
        
        audio_filename = f"{idx:03d}_{safe_title}"
        audio_path = os.path.join(audio_dir, audio_filename)
        
        # Download audio
        print("  ⬇ Downloading audio...")
        if not download_audio(video_id, audio_path):
            failed += 1
            failed_videos.append({'title': title, 'url': f'https://www.youtube.com/watch?v={video_id}'})
            continue
        
        # Find the actual audio file (yt-dlp adds extension)
        audio_file = None
        for ext in ['.mp3', '.m4a', '.webm']:
            potential_file = audio_path + ext
            if os.path.exists(potential_file):
                audio_file = potential_file
                break
        
        if not audio_file:
            print("  ✗ Audio file not found")
            failed += 1
            failed_videos.append({'title': title, 'url': f'https://www.youtube.com/watch?v={video_id}'})
            continue
        
        # Transcribe
        transcript_text, segments = transcribe_audio(audio_file, whisper_model)
        
        if transcript_text:
            # Save as plain text (for RAG)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Video: {title}\n")
                f.write(f"Video ID: {video_id}\n")
                f.write(f"URL: https://www.youtube.com/watch?v={video_id}\n")
                f.write("="*80 + "\n\n")
                f.write(transcript_text)
            
            # Save detailed transcript with timestamps (JSON)
            json_filename = f"{idx:03d}_{safe_title}.json"
            json_path = os.path.join(output_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'title': title,
                    'video_id': video_id,
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'transcript': transcript_text,
                    'segments': segments
                }, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Saved transcript ({len(transcript_text)} characters)")
            successful += 1
            
            # Optional: Delete audio file to save space
            # Uncomment the line below if you want to delete audio after transcription
            # os.remove(audio_file)
        else:
            failed += 1
            failed_videos.append({'title': title, 'url': f'https://www.youtube.com/watch?v={video_id}'})
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successfully processed: {successful}/{len(videos)}")
    print(f"Failed: {failed}/{len(videos)}")
    print(f"Skipped (private): {skipped}/{len(videos)}")
    
    if failed_videos:
        print(f"\nFailed videos:")
        for video in failed_videos[:5]:
            print(f"  - {video['title']}")
        if len(failed_videos) > 5:
            print(f"  ... and {len(failed_videos) - 5} more")
    
    print(f"\nTranscripts saved to '{output_dir}/' directory")
    print(f"Audio files saved to '{audio_dir}/' directory")

def main():
    print("Loading Whisper model...")
    print(f"Using '{WHISPER_MODEL}' model. This may take a moment on first run...")
    
    try:
        # Load Whisper model (downloads on first use)
        model = whisper.load_model(WHISPER_MODEL)
        print("✓ Model loaded successfully!\n")
        
        videos = get_playlist_videos(PLAYLIST_URL)
        print(f"Found {len(videos)} videos in playlist")
        
        # Ask user if they want to process all or just a few
        response = input(f"\nProcess all {len(videos)} videos? This will take a while. (y/n): ")
        if response.lower() != 'y':
            try:
                num = int(input("How many videos do you want to process? "))
                videos = videos[:num]
            except:
                print("Invalid input, processing first 5 videos")
                videos = videos[:5]
        
        process_videos(videos, OUTPUT_DIR, AUDIO_DIR, model)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nMake sure you have installed:")
        print("pip install openai-whisper yt-dlp")
        print("\nAnd install ffmpeg:")
        print("Mac: brew install ffmpeg")
        print("Ubuntu: sudo apt install ffmpeg")

if __name__ == "__main__":
    main()