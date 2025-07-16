import os
from youtube_transcript_api import YouTubeTranscriptApi

def get_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    if 'v=' in url:
        return url.split('v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0]
    else:
        return None

def main():
    links_file = 'youtubeLinks.txt'
    output_file = 'transcripts.txt'
    if not os.path.exists(links_file):
        print(f"{links_file} not found.")
        return
    with open(links_file, 'r') as f:
        links = [line.strip() for line in f if line.strip()]
    all_transcripts = []
    for link in links:
        video_id = get_video_id(link)
        if not video_id:
            print(f"Could not extract video ID from: {link}")
            continue
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = ' '.join([entry['text'] for entry in transcript])
            all_transcripts.append(f"--- Transcript for {link} ---\n{text}\n")
            print(f"Extracted transcript for {link}")
        except Exception as e:
            print(f"Failed to get transcript for {link}: {e}")
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write('\n'.join(all_transcripts))
    print(f"All transcripts written to {output_file}")

if __name__ == "__main__":
    main() 