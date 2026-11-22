"""KB ingestion agent: documents in, searchable chunks out.

Parse with Docling (OCR only for scanned PDFs) -> the LLM picks a chunking strategy ->
embed -> upsert into the vector index. The data between steps lives in this process, never in
the model's context, so a prompt cannot redirect a step to another file or index.
"""

from __future__ import annotations

import hashlib
import os
import time

from agentkit.contract import AgentCard, AgentError, Completed, Skill
from agentkit.docparse import parse
from agentkit.llm import get_llm
from agentkit.redis_conn import get_redis
from agentkit.server import JobContext, create_agent_app
from agentkit.vectors import embed, get_store
from agents.kb_common import COLLECTION, equipment_codes, save_doc
from agents.kb_ingestion.chunking import choose_strategy, make_chunks

CARD = AgentCard(
    name="kb-ingestion-agent",
    title="Knowledge-base ingestion",
    description="Adds documents (PDF, Word, PowerPoint, Excel, HTML, Markdown, scanned PDFs) to the "
                "searchable knowledge base. Use when the user uploads or names a document and asks to "
                "add, ingest, index or learn it.",
    version="1.0.0",
    url=os.environ.get("KB_INGESTION_URL", "http://localhost:8101"),
    skills=[Skill(id="ingest", name="Add a document to the knowledge base",
                  description="Parse, chunk, embed and index a document so it can be searched.",
                  requires_approval=True)],  # it writes to the shared knowledge base
    allowed_roles=["author"],
    document_input=True,
)


def handle(ctx: JobContext) -> Completed:
    if not ctx.request.files:
        raise AgentError("Attach or pick a document to add to the knowledge base.")
    parse_only = bool(ctx.parameters.get("parse_only"))
    store = get_store()
    reports, data = [], []
    for file_id in ctx.request.files:
        path = ctx.files.path(file_id, ctx.user)
        name = ctx.files.meta(file_id)["name"]
        started = time.time()
        ctx.progress("Parsing with Docling", name)
        doc = parse(path, name)
        if doc.ocr_used:
            ctx.progress("Scanned PDF detected: text read with Tesseract OCR", name)
        if parse_only:
            preview = doc.markdown[:3000]
            data.append({"file": name, "pages": len(doc.pages), "markdown": preview})
            reports.append(f"Parsed **{doc.title}** ({len(doc.pages)} pages). Nothing was indexed.")
            continue
        ctx.progress("Choosing a chunking strategy")
        strategy, reason = choose_strategy(get_llm(), doc)
        chunks = make_chunks(strategy, doc, embed)
        ctx.progress(f"Chunking: {strategy}", f"{len(chunks)} chunks. {reason}")
        doc_id = "doc-" + hashlib.sha1(path.read_bytes()).hexdigest()[:12]
        equipment = equipment_codes(doc.title + "\n" + doc.markdown)[:3]
        ctx.progress("Embedding and indexing", f"{len(chunks)} chunks")
        texts = [f"{doc.title} > {c.heading}\n{c.text}" for c in chunks]
        vectors = embed(texts)
        store.delete_doc(COLLECTION, doc_id)  # re-ingesting a file replaces its chunks
        store.upsert(
            COLLECTION,
            [f"{doc_id}:{i}" for i in range(len(chunks))],
            vectors,
            [{"doc_id": doc_id, "title": doc.title, "file_id": file_id, "file_name": name, "page": c.page,
              "heading": c.heading, "text": c.text, "order": i, "equipment": equipment}
             for i, c in enumerate(chunks)],
        )
        save_doc(doc_id, {"title": doc.title, "file_id": file_id, "file_name": name, "pages": len(doc.pages),
                          "chunks": len(chunks), "strategy": strategy, "ocr": doc.ocr_used,
                          "equipment": equipment, "owner": ctx.user, "indexed_at": time.time()})
        get_redis().set(f"kb:session:{ctx.session_id}", doc_id, ex=24 * 3600)
        took = time.time() - started
        ocr_note = " Read with OCR (scanned PDF)." if doc.ocr_used else ""
        reports.append(f"Added **{doc.title}**: {len(doc.pages)} pages, {len(chunks)} chunks, "
                       f"{strategy} chunking ({reason}).{ocr_note} Took {took:.0f} s.")
        data.append({"doc_id": doc_id, "title": doc.title, "chunks": len(chunks), "strategy": strategy,
                     "ocr": doc.ocr_used, "equipment": equipment})
    return Completed(message="\n\n".join(reports), data={"documents": data})


app = create_agent_app(CARD, handle)
