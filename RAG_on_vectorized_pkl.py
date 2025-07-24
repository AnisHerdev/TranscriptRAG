import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PKL_FILE = 'vectorized_chunks.pkl'
TOP_K = 3

def load_embeddings(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['model_name'], data['metadata']

def get_query_embedding(query, model_name):
    model = SentenceTransformer(model_name)
    return model.encode([query], convert_to_numpy=True)

def retrieve_top_k(query_embedding, embeddings, k=TOP_K):
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k_idx = np.argsort(similarities)[::-1][:k]
    return top_k_idx, similarities[top_k_idx]

def main():
    print(f"Loading embeddings from {PKL_FILE}...")
    embeddings, model_name, metadata = load_embeddings(PKL_FILE)
    print(f"Loaded {len(embeddings)} chunks. Using model: {model_name}")
    query = input("Enter your query: ")
    query_embedding = get_query_embedding(query, model_name)
    top_k_idx, top_k_scores = retrieve_top_k(query_embedding, embeddings, TOP_K)
    print(f"\nTop {TOP_K} relevant chunks:")
    for rank, (idx, score) in enumerate(zip(top_k_idx, top_k_scores), 1):
        meta = metadata[idx]
        print(f"\nRank {rank} (Score: {score:.4f}):")
        print(f"Video URL: {meta['video_url']}")
        print(f"Chunk Index: {meta['chunk_index']}")
        print(f"Chunk Text: {meta['chunk_text']}")

if __name__ == "__main__":
    main() 