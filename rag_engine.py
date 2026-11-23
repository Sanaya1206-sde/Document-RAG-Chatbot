import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('GROQ_API_KEY')
MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')
API_URL = 'https://api.groq.com/openai/v1/chat/completions'

SYSTEM_PROMPT = '''You are DocuMind, an assistant that helps people understand
the document(s) they uploaded. You are given numbered context passages taken
from those documents.
Rules:
1. Base every answer on the context passages. Never use outside facts about
   what the document says.
2. You MAY reason over the content. For example, if the document is a resume
   and the user asks which jobs fit it, analyse the skills, education and
   experience in the context and suggest suitable roles, explaining why.
3. If the question has nothing to do with the uploaded documents, or the
   context truly has no relevant information, reply exactly:
   I could not find this in the uploaded documents.
4. Cite the passages you used with their numbers, like [1] or [2].
5. Be clear, well structured and concise. Use short bullet points when helpful.'''


def call_llm(messages):
    if not API_KEY:
        raise RuntimeError('GROQ_API_KEY not found. Check your .env file.')
    headers = {
        'Authorization': 'Bearer ' + API_KEY,
        'Content-Type': 'application/json',
    }
    payload = {
        'model': MODEL,
        'messages': messages,
        'temperature': 0.2,
        'max_tokens': 1500,
        'reasoning_effort': 'low',
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f'LLM API error {response.status_code}: {response.text}')
    return response.json()['choices'][0]['message']['content']


def answer_question(question, store, top_k=6, session_id=None,
                    source_filter=None, history=None):
    """Answer a question using only the chunks that belong to one chat session.

    history: optional list of earlier {'role', 'content'} messages so that
    follow-up questions ("what about the second one?") make sense.
    """
    start = time.time()
    hits = store.search(
        question, top_k=top_k,
        session_id=session_id, source_filter=source_filter,
    )

    if not hits:
        return {
            'answer': 'No documents found in this chat. Please upload and process a document first.',
            'sources': [],
            'latency': time.time() - start,
        }

    context_parts = []
    for number, hit in enumerate(hits, start=1):
        source = hit['source']
        page = hit['page']
        text = hit['text']
        context_parts.append(f'[{number}] (Source: {source}, Page {page})\n{text}')
    context = '\n\n'.join(context_parts)

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    if history:
        for m in history[-6:]:
            messages.append({'role': m['role'], 'content': m['content']})
    messages.append({
        'role': 'user',
        'content': f'CONTEXT:\n{context}\n\nQUESTION: {question}',
    })

    answer = call_llm(messages)
    return {
        'answer': answer,
        'sources': hits,
        'latency': time.time() - start,
    }


if __name__ == '__main__':
    import sys
    from vector_store import VectorStore
    if len(sys.argv) < 2:
        print('Usage: python rag_engine.py "your question"')
        sys.exit(1)
    store = VectorStore()
    result = answer_question(' '.join(sys.argv[1:]), store, session_id='cli')
    print('\nANSWER:\n' + result['answer'])
    print('\nSOURCES:')
    for number, hit in enumerate(result['sources'], start=1):
        source = hit['source']
        page = hit['page']
        print(f'[{number}] {source}, page {page}')
    latency = result['latency']
    print(f'\nResponse time: {latency:.2f} seconds')