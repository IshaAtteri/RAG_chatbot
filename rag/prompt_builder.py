from __future__ import annotations

from typing import List
from .retriever import RetrievedChunk


def format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a readable context block
    that we can insert into the LLM prompt.
    """
    lines = []
    for i, ch in enumerate(chunks, start=1):
        header = f"[Source {i}: {ch.path} (chunk {ch.chunk_id}, score={ch.score:.2f})]"
        lines.append(header)
        lines.append(ch.text)
        lines.append("")  # blank line between chunks
    return "\n".join(lines).strip()


def build_rag_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    """
    Build a single string prompt for non-chat generation.

    We'll also make a chat-style builder later for create_chat_completion().
    """
    context_block = format_context(chunks)

    prompt = f"""You are a helpful assistant that answers questions using ONLY the context provided below.
If the answer is not clearly supported by the context, say you do not know.

Context:
{context_block}

Question:
{query}

Answer:"""

    return prompt.strip()


def build_chat_messages(query: str, chunks: List[RetrievedChunk]):
    context_block = format_context(chunks)

    system_content = (
        "You are a helpful assistant that answers questions using ONLY the provided context.\n"
        "If the context is insufficient or unrelated, reply that you do not know.\n"
        "Do NOT add sign-offs, author names, phrases like 'Written by [Your Name]', "
        "blog-style footers, or any meta commentary about scores or the prompt.\n"
        "Just answer the question directly and concisely in plain text."
    )

    user_content = f"""Context:
{context_block}

Question:
{query}

Important:
- Do NOT add any phrases like 'Written by [Your Name]'.
- Do NOT add 'Note:' sections or explanations about how the answer was generated.
- Respond ONLY with the answer itself, no extra commentary.
"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages
