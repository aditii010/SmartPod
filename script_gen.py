from concurrent.futures import ThreadPoolExecutor
from core.services.llm import chat_llm
import time

SYSTEM_PROMPT = """You are an expert educational podcast script writer. Create natural, engaging dialogue between a Tutor (experienced teacher) and a Student (curious learner).

Guidelines:
- Make it conversational and natural
- Use filler words occasionally ("you know", "um", "well", "so")
- Student should ask clarifying questions
- Tutor should explain clearly with examples
- Keep exchanges brief (2-4 turns per chunk)
- Show genuine curiosity and understanding
- Use casual language, not formal lecture style

Format your response EXACTLY like this:
Tutor: [text here]
Student: [text here]
Tutor: [text here]
Student: [text here]
"""


def generate_one_genai(chunk: str, index: int):
    """
    Generate script using REAL GenAI (Ollama)
    """
    try:
        user_prompt = f"""Create a natural dialogue about this content. Make it engaging and conversational:

CONTENT:
{chunk[:600]}

Create 3-5 dialogue exchanges between Tutor and Student. Format each line as:
Tutor: [dialogue]
Student: [dialogue]

Keep it natural and conversational!"""

        print(f"\n📝 Generating GenAI script for chunk {index}...")
        start_time = time.time()

        # Call Ollama for REAL AI generation
        response = chat_llm(SYSTEM_PROMPT, user_prompt, model="llama2")

        elapsed = time.time() - start_time
        print(f"   ✅ Generated in {elapsed:.1f}s")

        # Parse response into segments
        segments = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            if ':' in line and (line.startswith('Tutor:') or line.startswith('Student:')):
                parts = line.split(':', 1)
                speaker = parts[0].strip()
                text = parts[1].strip()

                if text:
                    segments.append({
                        "speaker": speaker,
                        "text": text,
                        "chunk_index": index
                    })

        if not segments:
            # Fallback if parsing fails
            segments = [{
                "speaker": "Tutor",
                "text": chunk[:300],
                "chunk_index": index
            }]

        print(f"   📊 Created {len(segments)} dialogue segments")
        return segments

    except Exception as e:
        print(f"❌ GenAI error for chunk {index}: {e}")
        return [{
            "speaker": "Tutor",
            "text": chunk[:300],
            "chunk_index": index
        }]


def generate_script_parallel(chunks):
    """
    Generate scripts using GenAI in parallel
    """
    print(f"\n{'=' * 60}")
    print(f"🤖 GENERATING GENAI DIALOGUE")
    print(f"Using: Ollama (llama2)")
    print(f"Chunks: {len(chunks)}")
    print(f"{'=' * 60}")

    # Limit chunks for speed (GenAI is slower)
    limited_chunks = chunks[:5]

    all_segments = []

    # Process sequentially for stability (Ollama can be memory-intensive)
    for i, chunk in enumerate(limited_chunks):
        segments = generate_one_genai(chunk, i)
        all_segments.extend(segments)

    all_segments.sort(key=lambda x: x.get('chunk_index', 0))

    print(f"\n✅ Total segments generated: {len(all_segments)}")
    print(f"{'=' * 60}\n")

    return all_segments
