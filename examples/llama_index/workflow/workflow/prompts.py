# Prompt to transform user query to the web search query format
TRANSFORM_QUERY_TEMPLATE = """\
Your task is to refine a query to ensure it is highly effective for retrieving relevant search results.
Analyze the given input to grasp the core semantic intent or meaning.

Original Query:
-------------------
{query}

Your goal is to rephrase or enhance this query to improve its search performance. Ensure the revised query is concise and directly aligned with the intended search objective.
Respond with the optimized query only"""


# Prompt for the final LLM query
FINAL_QUERY_TEMPLATE = """\
Your task is to answer the user query as a professional and very helpful assistant. You must use the given context for answering the question and not prior knowledge. Context information is below.

Context:
-------------------
{context}

User Question:
--------------
{query}

Respond with the answer to the question only:"""
