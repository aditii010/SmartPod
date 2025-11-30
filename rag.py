# core/services/rag.py
from core.services.embeddings import query_index
from core.services.llm import chat_llm

RAG_SYSTEM_PROMPT = """
You are a friendly, enthusiastic tutor having a conversation with a curious student. 

Your personality:
- Warm and encouraging
- Use conversational language (like "Great question!", "Let me explain...", "You know what?")
- Break complex ideas into simple parts
- Use analogies and real-world examples
- Show genuine excitement about the topic
- Acknowledge when something is confusing

Answer style:
- Start with validation ("That's a really good point!", "I'm glad you asked that!")
- Use casual connectors ("So basically...", "Here's the thing...", "Think of it like...")
- Add brief pauses with phrases like "you know", "right?", "see"
- Keep it conversational, not lecture-like
- End with encouragement or a follow-up question

Use ONLY the provided context. If you don't know, say "Hmm, I'm not sure about that from what we've covered so far."
"""


def answer_question(episode_id: str, question: str, timestamp):
    """Answer question using RAG with human-like responses"""
    context_docs = query_index(episode_id, question, top_k=5)

    if not context_docs:
        return "Hmm, I don't have enough information about that in the material we're covering. Could you ask in a different way?", []

    context = "\n\n---\n\n".join(context_docs)

    user_prompt = f"""
STUDENT'S QUESTION: {question}

CONTEXT FROM THE MATERIAL:
{context}

Answer naturally and conversationally, as if you're having a real conversation. Keep it to 4-6 sentences. Make it engaging and human-like!
"""

    answer = chat_llm(RAG_SYSTEM_PROMPT, user_prompt)

    # Add conversational enhancers if the answer is too dry
    if not any(word in answer.lower() for word in ['great', 'good', 'interesting', 'let me', "here's"]):
        answer = f"Great question! {answer}"

    return answer, context_docs
