from agent.prompt import SYSTEM_PROMPT

def answer(query, embed_fn, llm, store):
    # Embed query (single vector)
    query_embedding = embed_fn(query)

    # Retrieve contexts
    contexts = store.search(query_embedding, k=5)

    # Vertex AI–like guard
    if not contexts:
        return "I don’t have that information in the uploaded documents."

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{''.join(contexts)}

Question:
{query}
"""

    return llm.generate(prompt)
