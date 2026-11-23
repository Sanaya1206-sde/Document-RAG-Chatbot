# ingest_folder.py
# Loads every PDF, DOCX and TXT file in a folder (and its subfolders).

import os
import sys

from document_processor import process_file
from vector_store import VectorStore

SUPPORTED = ('.pdf', '.docx', '.txt')

if len(sys.argv) < 2:
    print('Usage: python ingest_folder.py <folder_path>')
    sys.exit(1)

folder = sys.argv[1]
store = VectorStore()
total = 0

for root, dirs, files in os.walk(folder):
    for name in files:
        if not name.lower().endswith(SUPPORTED):
            continue
        path = os.path.join(root, name)
        # Use the path inside the folder as the name, so two files with
        # the same name in different subjects don't overwrite each other.
        label = os.path.relpath(path, folder)
        try:
            count = store.add_chunks(process_file(path, source_name=label))
            total += count
            print('OK   ', label, '-', count, 'chunks')
        except Exception as e:
            print('SKIP ', label, '-', e)

print('Done. Total chunks stored:', total)