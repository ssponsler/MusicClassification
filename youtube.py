from googleapiclient.discovery import build
import yt_dlp
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the API key
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Ensure the API key is available
if not YOUTUBE_API_KEY:
    raise ValueError("ERROR: YOUTUBE_API_KEY is missing. Add it to your .env file.")

print(f"Using API Key: {YOUTUBE_API_KEY[:5]}**** (Loaded Successfully)")

# Initialize YouTube API client
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Genres to collect
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Directory to store downloaded songs
save_dir = "music_samples"
os.makedirs(save_dir, exist_ok=True)

def get_youtube_video(query):
    """Search for a YouTube video for the given genre and return its URL."""
    try:
        request = youtube.search().list(
            q=query, 
            part="snippet", 
            type="video", 
            maxResults=3  # get up to 3 results
        )
        response = request.execute()
        
        # Return the first valid video URL
        for item in response["items"]:
            video_url = "https://www.youtube.com/watch?v=" + item["id"]["videoId"]
            return video_url
        
    except Exception as e:
        print(f"⚠️ ERROR: Failed to fetch YouTube video for {query} - {e}")
    
    return None

def download_audio(video_url, genre):
    """Download the audio from a YouTube video and save as a WAV file."""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{save_dir}/{genre}/%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav'
                # 'preferredquality': '192'  # Not needed for WAV
            }]
        }
        
        os.makedirs(f"{save_dir}/{genre}", exist_ok=True)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    
    except Exception as e:
        print(f"ERROR: Failed to download {video_url} - {e}")

# Loop through genres and download music
for genre in genres:
    print(f"\n🔍 Searching for {genre} music...")
    video_url = get_youtube_video(f"{genre} music")
    
    if video_url:
        print(f"🎵 Downloading WAV: {video_url}")
        download_audio(video_url, genre)
    else:
        print(f"No valid videos found for {genre}")

print("\nDownload complete!")
