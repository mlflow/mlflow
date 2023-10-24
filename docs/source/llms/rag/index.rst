====================================
Retrieval Augmented Generation (RAG)
====================================

Retrieval Augmented Generation, or RAG, is a powerful and efficient approach to natural 
language processing that combines the strength of both pre-trained foundation models and 
retrieval mechanisms. It allows the generative model to access a dataset of documents 
through a retrieval mechanism, which enhances generated responses to be more contextually relevant
and factually accurate. This improvement results in a cost-effective and accessible alternative
to training custom models for specific use cases.

The Retrieval mechanism works by embedding documents and questions in the same latent space, allowing
a user to ask a question and get the most relevant document chunk as a response. This mechanism then passes
the contextual chunk to the generative model, resulting in better quality responses with fewer hallucinations.

Benefits of RAG
===============
1. Provides LLM access to external knowledge through documents, resulting in contextually accurate and 
factual responses.
2. RAG is more cost-effective than fine-tuning, since it doesn't require the labeled data and computational
resources that come with model training.

.. toctree::
    :maxdepth: 1
    
    Full Notebooks <notebooks/index>