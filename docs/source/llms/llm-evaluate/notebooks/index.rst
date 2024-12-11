LLM Evaluation Examples
=======================

The notebooks listed below contain step-by-step tutorials on how to use MLflow to evaluate LLMs. 
The first notebook is centered around evaluating an LLM for question-answering with a 
prompt engineering approach. The second notebook is centered around evaluating a RAG system. 
Both notebooks will demonstrate how to use MLflow's builtin metrics such as token_count and 
toxicity as well as LLM-judged intelligent metrics such as answer_relevance. The third notebook 
is the same as the second notebook, but uses Databricks's served llama2-70b as the judge instead 
of gpt-4.

.. toctree::
    :maxdepth: 1
    :hidden:

    question-answering-evaluation.ipynb
    rag-evaluation.ipynb
    rag-evaluation-llama2.ipynb
    huggingface-evaluation.ipynb

QA Evaluation Tutorial
----------------------

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="question-answering-evaluation.html">
                    <div class="header">
                        LLM Question Answering Evaluation with MLflow
                    </div>
                    <p>
                        Learn how to evaluate various LLMs and RAG systems with MLflow, leveraging simple metrics such as toxicity, as well as LLM-judged metrics as relevance, and even custom LLM-judged metrics such as professionalism.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="huggingface-evaluation.html">
                    <div class="header">
                        Evaluating a 🤗 Hugging Face LLMs with MLflow
                    </div>
                    <p>
                        Learn how to evaluate various Open-Source LLMs available in Hugging Face, leveraging MLflow's built-in LLM metrics and experiment tracking to manage models and evaluation results.
                    </p>
                </a>
            </div>
        </article>
    </section>

RAG Evaluation Tutorials
------------------------

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="rag-evaluation.html">
                    <div class="header">
                        RAG Evaluation with MLflow and GPT-4 as Judge
                    </div>
                    <p>
                        Learn how to evaluate RAG systems with MLflow, leveraging <b>OpenAI GPT-4</b> model as a judge.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="rag-evaluation-llama2.html">
                    <div class="header">
                        RAG Evaluation with MLflow and Llama-2-70B as Judge
                    </div>
                    <p>
                        Learn how to evaluate RAG systems with MLflow, leveraging <b>Llama 2 70B model</b> hosted on Databricks serving endpoint.
                    </p>
                </a>
            </div>
      </article>
    </section>
