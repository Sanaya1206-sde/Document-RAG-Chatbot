import json
import os
import tempfile
import time
import uuid

import streamlit as st

from document_processor import process_file
from rag_engine import answer_question
from vector_store import VectorStore

CHATS_FILE = 'chats.json'

st.set_page_config(page_title='DocuMind', page_icon='📄', layout='wide')

# ---------------- Styling ----------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 900px; }
    .dm-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .dm-sub { color: #9aa0a6; margin-top: 0; margin-bottom: 1rem; }
    .dm-chip {
        display: inline-block; padding: 4px 12px; margin: 0 6px 6px 0;
        border-radius: 999px; background: rgba(255, 75, 75, 0.15);
        border: 1px solid rgba(255, 75, 75, 0.4); font-size: 0.85rem;
    }
    .dm-welcome {
        text-align: center; padding: 3rem 1rem; border: 1px dashed #444;
        border-radius: 16px; margin-top: 1rem;
    }
    .dm-welcome h2 { margin-bottom: 0.3rem; }
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left; justify-content: flex-start;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_store():
    return VectorStore()


store = load_store()


# ---------------- Chat history (saved to chats.json) ----------------
def load_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_chats():
    with open(CHATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.chats, f, ensure_ascii=False, indent=2)


def new_chat():
    chat_id = uuid.uuid4().hex[:10]
    st.session_state.chats[chat_id] = {
        'id': chat_id,
        'title': 'New chat',
        'created': time.time(),
        'documents': [],
        'messages': [],
    }
    st.session_state.active = chat_id
    save_chats()


def active_chat():
    return st.session_state.chats[st.session_state.active]


if 'chats' not in st.session_state:
    st.session_state.chats = load_chats()
if 'upload_round' not in st.session_state:
    st.session_state.upload_round = 0
if 'active' not in st.session_state or st.session_state.active not in st.session_state.chats:
    if st.session_state.chats:
        newest = max(st.session_state.chats.values(), key=lambda c: c['created'])
        st.session_state.active = newest['id']
    else:
        new_chat()


def show_sources(sources):
    with st.expander('📚 Sources used'):
        for number, s in enumerate(sources, start=1):
            st.markdown(f"**[{number}] {s['source']}, page {s['page']}** (similarity {s['score']:.2f})")
            st.caption(s['text'][:400] + '...')


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown('### 📄 DocuMind')

    if st.button('➕ New chat', use_container_width=True, type='primary'):
        # Reuse the current chat if it is still empty
        if active_chat()['messages'] or active_chat()['documents']:
            new_chat()
            st.session_state.upload_round += 1
        st.rerun()

    st.markdown('##### Upload to this chat')
    uploaded_files = st.file_uploader(
        'PDF, DOCX or TXT',
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.active}_{st.session_state.upload_round}",
        label_visibility='collapsed',
    )

    if st.button('Process documents', use_container_width=True):
        if not uploaded_files:
            st.warning('Please choose at least one file first.')
        else:
            chat = active_chat()
            for f in uploaded_files:
                with st.spinner('Processing ' + f.name + ' ...'):
                    suffix = os.path.splitext(f.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(f.getbuffer())
                        tmp_path = tmp.name
                    try:
                        chunks = process_file(tmp_path, source_name=f.name)
                        store.add_chunks(chunks, session_id=chat['id'])
                        if f.name not in chat['documents']:
                            chat['documents'].append(f.name)
                        if chat['title'] == 'New chat':
                            chat['title'] = f.name
                    except Exception as e:
                        st.error(f.name + ': ' + str(e))
                    finally:
                        os.remove(tmp_path)
            save_chats()
            st.session_state.upload_round += 1
            st.rerun()

    st.divider()
    st.markdown('##### History')
    chats_sorted = sorted(
        st.session_state.chats.values(), key=lambda c: c['created'], reverse=True
    )
    for c in chats_sorted:
        is_active = c['id'] == st.session_state.active
        col_a, col_b = st.columns([5, 1])
        label = ('● ' if is_active else '') + c['title']
        if col_a.button(label, key='open_' + c['id'], use_container_width=True):
            st.session_state.active = c['id']
            st.rerun()
        if col_b.button('🗑', key='del_' + c['id'], help='Delete this chat'):
            store.delete_session(c['id'])
            del st.session_state.chats[c['id']]
            save_chats()
            if st.session_state.active == c['id']:
                if st.session_state.chats:
                    newest = max(st.session_state.chats.values(), key=lambda x: x['created'])
                    st.session_state.active = newest['id']
                else:
                    new_chat()
            st.rerun()

# ---------------- Main area ----------------
chat = active_chat()
has_docs = len(chat['documents']) > 0

st.markdown('<p class="dm-title">📄 DocuMind</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="dm-sub">Ask questions about your own documents. Every answer shows its sources.</p>',
    unsafe_allow_html=True,
)

if has_docs:
    chips = ''.join(f'<span class="dm-chip">📎 {d}</span>' for d in chat['documents'])
    st.markdown(chips, unsafe_allow_html=True)

if not chat['messages']:
    if has_docs:
        st.info('Document ready. Ask your first question below.')
    else:
        st.markdown(
            """
            <div class="dm-welcome">
                <h2>Welcome to DocuMind</h2>
                <p>Upload a PDF, DOCX or TXT file from the sidebar and click
                <b>Process documents</b>.<br>
                Answers will come only from the documents in this chat.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

for m in chat['messages']:
    with st.chat_message(m['role']):
        st.markdown(m['content'])
        if m.get('latency') is not None:
            st.caption(f"Response time: {m['latency']:.2f}s")
        if m.get('sources'):
            show_sources(m['sources'])

question = st.chat_input(
    'Ask a question about your document' if has_docs else 'Upload a document to start chatting',
    disabled=not has_docs,
)

if question:
    history = [{'role': m['role'], 'content': m['content']} for m in chat['messages']]
    chat['messages'].append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        with st.spinner('Searching your document...'):
            try:
                result = answer_question(
                    question, store,
                    session_id=chat['id'],
                    history=history,
                )
                st.markdown(result['answer'])
                st.caption(f"Response time: {result['latency']:.2f}s")
                if result['sources']:
                    show_sources(result['sources'])
                chat['messages'].append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'sources': result['sources'],
                    'latency': result['latency'],
                })
                save_chats()
            except Exception as e:
                st.error('Something went wrong: ' + str(e))