# TranscriptRAG: YouTube Transcript RAG Chatbot

TranscriptRAG is a simple Retrieval-Augmented Generation (RAG) chatbot that leverages transcripts from YouTube videos and a GPT-2 language model to answer user questions. It extracts transcripts from a list of YouTube links, indexes them for retrieval, and uses GPT-2 to generate context-aware responses.

## Features
- Extracts transcripts from YouTube videos using `youtube_transcript_api`.
- Stores all transcripts in a single file for easy processing.
- Implements a basic RAG pipeline: retrieves relevant transcript chunks using TF-IDF and Nearest Neighbors, then generates answers with GPT-2.
- Simple command-line chatbot interface.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/AnisHerdev/TranscriptRAG.git
cd TranscriptRAG
```

### 2. Create a Python 3.11 Virtual Environment
It is recommended to use Python 3.11 for compatibility.

#### On Windows:
```bash
py -3.11 -m venv myenv
myenv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3.11 -m venv .venv --prompt TranscriptRAG
source .venv/bin/activate
```

### 3. Install Dependencies
With your virtual environment activated, run:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare YouTube Links
- Create a file named `youtubeLinks.txt` in the project root.
- Add one YouTube video link per line.

Example `youtubeLinks.txt`:
```
https://www.youtube.com/watch?v=example1
https://youtu.be/example2
```

### 2. Extract Transcripts
Run the following command to extract transcripts from all listed videos:
```bash
python extract_transcripts.py
```
- This will create (or overwrite) a file named `transcripts.txt` containing all transcripts.

### 3. Start the RAG Chatbot
Run the chatbot with:
```bash
python applyRAG.py
```
- Type your questions at the prompt. The bot will retrieve relevant transcript chunks and generate an answer using GPT-2.
- Type `exit` or `quit` to stop the chatbot.

## Notes
- The first run of `applyRAG.py` will download the GPT-2 model from HuggingFace. Ensure you have an internet connection.
- Only videos with available transcripts (auto-generated or manual) will be processed.
- For best results, use clear and concise questions.

## Credits
- [youtube_transcript_api](https://github.com/jdepoix/youtube-transcript-api)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [scikit-learn](https://scikit-learn.org/)

---

Feel free to contribute or open issues on GitHub! 