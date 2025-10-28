"""
Video ID extractor for YouTube URLs
"""

import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from various forms of YouTube URLs
    
    Supports formats like:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://youtube.com/shorts/VIDEO_ID
    """
    if not url:
        return None
        
    # Clean the URL
    url = url.strip()
    
    # Try to extract from standard YouTube URL
    parsed = urlparse(url)
    if 'youtube.com' in parsed.netloc:
        if 'watch' in parsed.path:
            # Standard watch URL
            query = parse_qs(parsed.query)
            return query.get('v', [None])[0]
        elif 'shorts' in parsed.path:
            # YouTube shorts URL
            return parsed.path.split('/')[-1]
    elif 'youtu.be' in parsed.netloc:
        # Short URL format
        return parsed.path.lstrip('/')
        
    # Try to extract directly if it looks like a video ID
    if re.match(r'^[A-Za-z0-9_-]{11}$', url):
        return url
        
    return None

def main():
    """Interactive mode for testing video ID extraction"""
    print("YouTube Video ID Extractor")
    print("=" * 50)
    print("Enter YouTube URLs (one per line)")
    print("Enter a blank line to finish")
    print()
    
    while True:
        url = input("Enter YouTube URL: ").strip()
        if not url:
            break
            
        video_id = extract_video_id(url)
        if video_id:
            print(f"Video ID: {video_id}")
        else:
            print("Could not extract video ID from URL")
        print()

if __name__ == "__main__":
    main()