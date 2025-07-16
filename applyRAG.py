import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

TRANSCRIPTS_FILE = 'transcripts.txt'
CHUNK_SIZE = 300  # Number of words per chunk
TOP_K = 3  # Number of relevant chunks to retrieve

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def load_transcripts():
    if not os.path.exists(TRANSCRIPTS_FILE):
        raise FileNotFoundError(f"{TRANSCRIPTS_FILE} not found.")
    with open(TRANSCRIPTS_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    return chunk_text(text)

def build_retriever(chunks):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(chunks)
    nn = NearestNeighbors(n_neighbors=TOP_K, metric='cosine').fit(X)
    return vectorizer, nn

def retrieve(query, vectorizer, nn, chunks):
    q_vec = vectorizer.transform([query])
    dists, idxs = nn.kneighbors(q_vec)
    return [chunks[i] for i in idxs[0]]

def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    return tokenizer, model

def generate_response(context, query, tokenizer, model, max_length=150):
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
    outputs = model.generate(inputs, max_length=inputs.shape[1]+max_length, do_sample=True, top_p=0.95, top_k=50, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return answer.strip()

def main():
    print("Loading transcripts and building retriever...")
    chunks = load_transcripts()
    vectorizer, nn = build_retriever(chunks)
    tokenizer, model = load_gpt2()
    print("RAG chatbot ready. Type your question (or 'exit' to quit):")
    while True:
        query = input("You: ")
        if query.lower() in ('exit', 'quit'): break
        relevant_chunks = retrieve(query, vectorizer, nn, chunks)
        context = '\n'.join(relevant_chunks)
        response = generate_response(context, query, tokenizer, model)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main() 