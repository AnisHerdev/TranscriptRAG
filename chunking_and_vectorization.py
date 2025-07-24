import os
import re
import pickle
from sentence_transformers import SentenceTransformer
import nltk

TRANSCRIPTS_FILE = 'transcripts.txt'
SENTENCES_PER_CHUNK = 5  # Number of sentences per chunk
CHUNK_STRIDE = 2         # Overlap stride (number of sentences to move for next chunk)
OUTPUT_FILE = 'vectorized_chunks.pkl'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Ensure NLTK punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_transcripts(filepath):
    """Extracts (video_url, transcript) pairs from the transcripts.txt file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split on transcript headers
    entries = re.split(r'^--- Transcript for (.+?) ---$', content, flags=re.MULTILINE)
    transcripts = []
    # entries[0] is preamble (empty), then alternating url, text
    for i in range(1, len(entries), 2):
        url = entries[i].strip()
        text = entries[i+1].strip()
        if url and text:
            transcripts.append((url, text))
    return transcripts


def chunk_sentences(sentences, sentences_per_chunk=SENTENCES_PER_CHUNK, stride=CHUNK_STRIDE):
    chunks = []
    for i in range(0, len(sentences), stride):
        chunk = sentences[i:i+sentences_per_chunk]
        if len(chunk) < 2:
            break
        chunks.append(' '.join(chunk))
        if i + sentences_per_chunk >= len(sentences):
            break
    return chunks


def main():
    if not os.path.exists(TRANSCRIPTS_FILE):
        print(f"{TRANSCRIPTS_FILE} not found.")
        return
    print("Extracting transcripts...")
    transcripts = extract_transcripts(TRANSCRIPTS_FILE)
    all_chunks = []
    chunk_metadata = []  # (video_url, chunk_index, chunk_text)
    for url, text in transcripts:
        sentences = nltk.sent_tokenize(text)
        chunks = chunk_sentences(sentences)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({'video_url': url, 'chunk_index': idx, 'chunk_text': chunk})
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Encoding chunks with transformer embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    print(f"Embeddings shape: {embeddings.shape}")
    # Save embeddings and metadata
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'model_name': EMBEDDING_MODEL, 'metadata': chunk_metadata}, f)
    print(f"Saved embeddings and metadata to {OUTPUT_FILE}")


if __name__ == "__main__":
    main() 