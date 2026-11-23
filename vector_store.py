import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, persist_dir='chroma_db', collection_name='documind'):
        # The first run downloads this small model (about 90 MB).
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={'hnsw:space': 'cosine'},
        )

    @staticmethod
    def _build_where(session_id=None, source_filter=None):
        conditions = []
        if session_id:
            conditions.append({'session_id': session_id})
        if source_filter:
            conditions.append({'source': source_filter})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {'$and': conditions}

    def add_chunks(self, chunks, session_id='default'):
        """Store chunks and tag each one with the chat session it belongs to."""
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
        ids = [f'{session_id}::{c.source}::{c.chunk_id}' for c in chunks]
        metadatas = [
            {'source': c.source, 'page': c.page_number, 'session_id': session_id}
            for c in chunks
        ]
        # upsert = add, or overwrite if the same id already exists
        self.collection.upsert(
            ids=ids, documents=texts,
            embeddings=embeddings, metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query, top_k=4, session_id=None, source_filter=None):
        where = self._build_where(session_id, source_filter)

        if where:
            matching = len(self.collection.get(where=where, include=[])['ids'])
        else:
            matching = self.collection.count()
        if matching == 0:
            return []

        query_embedding = self.model.encode([query]).tolist()
        args = {
            'query_embeddings': query_embedding,
            'n_results': min(top_k, matching),
        }
        if where:
            args['where'] = where
        results = self.collection.query(**args)

        hits = []
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        dists = results['distances'][0]
        for text, meta, dist in zip(docs, metas, dists):
            hits.append({
                'text': text,
                'source': meta['source'],
                'page': meta['page'],
                'score': 1 - dist,   # higher = more similar
            })
        return hits

    def list_documents(self, session_id=None):
        where = self._build_where(session_id)
        if where:
            data = self.collection.get(where=where, include=['metadatas'])
        else:
            data = self.collection.get(include=['metadatas'])
        names = {m['source'] for m in data['metadatas']}
        return sorted(names)

    def delete_session(self, session_id):
        """Remove every chunk that belongs to one chat."""
        self.collection.delete(where={'session_id': session_id})

    def clear(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_collection()


if __name__ == '__main__':
    import sys
    from document_processor import process_file
    if len(sys.argv) < 3:
        print('Usage: python vector_store.py <file> "your question"')
        sys.exit(1)
    store = VectorStore()
    count = store.add_chunks(process_file(sys.argv[1]), session_id='cli')
    print('Stored chunks:', count)
    for hit in store.search(sys.argv[2], top_k=3, session_id='cli'):
        score = hit['score']
        source = hit['source']
        page = hit['page']
        preview = hit['text'][:150].replace('\n', ' ')
        print(f'[{score:.2f}] {source} page {page}: {preview}...')