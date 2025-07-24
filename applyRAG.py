import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

TRANSCRIPTS_FILE = 'transcripts.txt'
CHUNK_SIZE = 300  # Number of words per chunk
TOP_K = 2  # Number of relevant chunks to retrieve

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
    # Define maximum tokens for context and query
    max_input_length = 1024  # GPT-2's max input length
    max_context_tokens = max_input_length - 100  # Reserve 100 tokens for query and prompt structure

    # Tokenize context and truncate to fit within max_context_tokens
    context_tokens = tokenizer.encode(context, return_tensors='pt', truncation=True, max_length=max_context_tokens)
    query_tokens = tokenizer.encode(query, return_tensors='pt', truncation=True, max_length=50)  # Limit query length

    # Construct prompt
    prompt = f"Context: {tokenizer.decode(context_tokens[0], skip_special_tokens=True)}\nQuestion: {tokenizer.decode(query_tokens[0], skip_special_tokens=True)}\nAnswer:"
    prompt_tokens = tokenizer.encode(prompt, return_tensors='pt')

    # Double-check token length
    if prompt_tokens.shape[1] > max_input_length:
        print("[Warning] Prompt still too long after truncation. Further truncating context.")
        # Re-truncate context to fit
        context_tokens = context_tokens[:, -(max_input_length - 100):]
        prompt = f"Context: {tokenizer.decode(context_tokens[0], skip_special_tokens=True)}\nQuestion: {tokenizer.decode(query_tokens[0], skip_special_tokens=True)}\nAnswer:"
        prompt_tokens = tokenizer.encode(prompt, return_tensors='pt')

    # Prepare attention mask
    attention_mask = torch.ones_like(prompt_tokens)

    # Ensure max_length for generation
    gen_max_length = prompt_tokens.shape[1] + max_length

    if torch.cuda.is_available():
        prompt_tokens = prompt_tokens.to('cuda')
        attention_mask = attention_mask.to('cuda')

    try:
        outputs = model.generate(
            prompt_tokens,
            attention_mask=attention_mask,
            max_length=gen_max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = outputs[0]
        gen_tokens = generated[prompt_tokens.shape[1]:]
        answer = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        if not answer.strip():
            answer = "[No answer generated. Try a shorter or different question.]"
        return answer.strip()
    except Exception as e:
        return f"[Error] Failed to generate response: {str(e)}"

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