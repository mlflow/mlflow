.. _retrieval-question-generation:

====================================
Question Generation for Retrieval Evaluation (Experimental)
====================================

MLFlow provides a state-of-the-art experience for build Retrieval-Augmented Generation (RAG) models. 
RAG is a cutting edge approach that combines the strengths of retrieval models and generative models.
It effectively merges the capabilities of searching and generating text to provide more contextually
relevant and coherent responses to questions. RAG leverges a retriever to find context documents, and
this novel approach has revolutionized various NLP tasks.

Naturally, we want to be able to evaluate this retriever system for the RAG model to compare and judge its
performance. To evaluate a retriever system, we would first need a test set of questions on the documents.
These questions need to be diverse, relevant, and coherent. Manually generating questions may be challenging
because it first requires you to understand the documents, and spend lots of time coming up with questions 
for them. 

We want to make this process simpler by utilizing an LLM to generate questions for this test set. This
tutorial will walk through how to generate the questions and how to analyze the diversity and relevance
of the questions.

.. _retrieval-question-generation-quickstart:

The following guide will walk you through generating the question test set for retriever evaluation.

Step 1: Install and Load Packages
==================================

.. code-block:: python
   import os
   import pickle
   from langchain.docstore.document import Document
   from langchain.embeddings import OpenAIEmbeddings
   import openai

Step 2: Load Data
==================
Load the data through Langchain Documents to utilize their embedding models.

.. code-block:: python
   from langchain.docstore.document import Document
   import pickle

   docs, metadatas = None, None
   with open("/dbfs/bbqiu/mlflow_docs.pkl", "rb") as f:
   docs = pickle.load(f)
   with open("/dbfs/bbqiu/mlflow_docs_metadata.pkl", "rb") as f:
   metadatas = pickle.load(f)

   documents = []
   for i in range(len(docs)):
   doc, mdata = docs[i], metadatas[i]
   documents.append(Document(page_content=doc, metadata=mdata))

   print(documents[0])

Step 3: Set OpenAI Key
======================
Set the OpenAI key to utilize embeddings and GPT.

.. code-block:: python
   openai.api_key = "<redacted>"

Step 4: Generate Questions
==========================
Generate the list of questions with GPT, utilizing prompt engineering to produce better quality questions.
We have found that better quality questions are produced if you ask it for multiple questions per chunk.
In addition, sometimes the question references the document without explaining what it is referencing.
Telling GPT explicitly to explain references helps with this.

.. code-block:: python
   question_per_chunk = 5
   queries = []
   chunks = []
   for doc in documents:
      chunk = doc.page_content
      chunks.append(chunk)
      params = {
         "model": "gpt-3.5-turbo",
         "messages": [
               {
                  "role": "user",
                  "content": f"{chunk}.\n Please generate {question_per_chunk} questions based on the above document. The questions should be diverse and ask for different aspects of the document. Don't give vague references to the document without description. Split each question with a newline"
               }
         ],
      }

      response = openai.ChatCompletion.create(**params)
      response_queries = response.choices[0].message.content
      for q in response_queries.splitlines():
         q = " ".join(q.split()[1:])
         print("\nquery:", q)
         queries.append({"query": q})

Step 5: Quality Analysis of Questions Generated (Optional)
==========================================================
If you would like to compare quality of questions generated across different prompts, we can
analyze the quality of questions manually and in aggregate. We want to evaluate questions 
along two dimensions - their diversity and relevance.

Diversity
---------
Diversity of questions is important because we want questions to cover the majority of the
document content. In addition, we want to be able to evaluate the retriever with different 
forms of questioning. We want to be able to have harder questions and easier questions. All
of these are not straightforward to analyze, and we decided to analyze its through question
length and latent space embeddings.

Length gives a sense of how diverse the questions are. Some questions may be wordy while
others are straight to the point. It also allows us to identify problems with the question
generated. For example, you may identify some questions to have a length of 0.

.. code-block:: python
   # Length
   queries_len = pd.DataFrame([len(query["query"]) for query in queries], columns=["length"])
   queries_len.hist(bins=100)
   plt.title("Query Length")
   plt.xlabel("Query Length")
   plt.ylabel("Frequency")
   plt.show()

In addition to visual representation, we also want to look at more concrete percentile values.

.. code-block:: python
   #Calculating percentile values
   p10 = int(queries_len["length"].quantile(0.10))
   p90 = int(queries_len["length"].quantile(0.90))
   print("p10-p90 range", p90-p10)

We noticed that the short queries are all empty strings, and hence we need to filter for this.

.. code-block:: python
   # Short queries are all empty strings, need to filter for this.
   [query["query"] for query in queries if len(query["query"]) < 5]

There are also a couple queries that are long. However, these seem fine.

.. code-block:: python
   # Long queries seem fine
   [query["query"] for query in queries if len(query["query"]) > 160]

Latent space embeddings contain semantic information about the question. This can be used to 
evaluate the diversity and the difference between two questions. To do so, we will need to map the
high dimensional space to a lower dimensional space. We utilize PCA and TSNE to map the embeddings 
into a 2-dimensional space for visualization.

.. code-block:: python
   # Need to post process to remove empty queries
   queries_list = [query["query"] for query in queries if len(query["query"]) != 0]

   embeddings = OpenAIEmbeddings()
   query_embeddings = embeddings.embed_documents(queries_list)
   pca = sklearn.decomposition.PCA(n_components=50)
   lower_dim_query_embeddings = pca.fit_transform(query_embeddings)

   benchmark_queries = ["What is MLFlow", "What is MlFlow about", "Tell me about MlFlow Tracking", "What are the benefits of using MlFlow", "How can I use spark in model registry"]
   benchmark_queries_embeddings = embeddings.embed_documents(benchmark_queries)
   lower_dim_benchmark_query_embeddings = pca.transform(benchmark_queries_embeddings)

   tsne = sklearn.manifold.TSNE(n_components=2)
   lower_dim_embeddings = tsne.fit_transform(np.concatenate([lower_dim_query_embeddings, lower_dim_benchmark_query_embeddings], axis=0))

To visualize the points in 2-dimensional space, we utilize a scatterplot.

.. code-block:: python
   labels = np.concatenate([np.zeros(len(lower_dim_query_embeddings)), np.ones(len(lower_dim_benchmark_query_embeddings))])
   data = pd.DataFrame(np.concatenate([lower_dim_embeddings, np.expand_dims(labels, axis=1)], axis=1), columns=["x", "y", "label"]) 

Document relevance
------------------

You can manually verify that the questions are relevant to their respective chunks here.

.. code-block:: python
   print(len(chunks), len(queries))
   # Manual checking of document relevance
   for i, chunk in enumerate(chunks[:question_per_chunk]):
      print(chunk)
      print(queries[i*question_per_chunk:i*question_per_chunk+question_per_chunk])
      print('-'*100)

We also define relevance through cosine similarity of embedding. However, just a cosine similarity score
is not interpretable without something to compare it to. Hence, we define relative question relevance as:
$$\frac{cossim(chunk_q)}{\frac{1}{len(chunks)-1}\sum_{i != q}cossim(chunk_{i})}$$

.. code-block:: python
   def cossim(x, y):
      return np.dot(x,y)/(np.linalg.norm(x) * np.linalg.norm(y))

   query_relevances = []
   for i, query in enumerate(embedded_queries):
      q = i // 5
      chunk_sim = cossim(query, embedded_chunks[q])
      other_chunk_sim = []
      for j, chunk in enumerate(embedded_chunks):
         if j != q:
            other_chunk_sim.append(cossim(query, chunk))
      query_relevances.append({"query": queries[i]["query"], "chunk": chunks[q], "score": chunk_sim / np.average(other_chunk_sim)})
   
   query_relevances

Visualizing the distribution of relative relevance score.

.. code-block:: python
   # Score above 1 means it is more relevant to its chunk than other chunks in the document (relative relevance). This shows that most chunks are relatively relevant.
   scores = [x["score"] for x in query_relevances]
   plt.hist(scores, bins=40)
