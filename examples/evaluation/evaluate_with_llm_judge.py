import logging
import os

import pandas as pd

from mlflow.metrics.base import EvaluationExample
from mlflow.metrics.utils.make_genai_metric import make_genai_metric

logging.getLogger("mlflow").setLevel(logging.ERROR)

# Uncomment the following lines to run this script without using a real OpenAI API key.
# os.environ["OPENAI_API_KEY"] = "test"

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."


# testing with OpenAI gpt-3.5-turbo
example = EvaluationExample(
    input="What is MLflow?",
    output="MLflow is an open-source platform for managing machine "
    "learning workflows, including experiment tracking, model packaging, "
    "versioning, and deployment, simplifying the ML lifecycle.",
    score=4,
    justification="The definition effectively explains what MLflow is "
    "its purpose, and its developer. It could be more concise for a 5-score.",
    variables={
        "ground_truth": "MLflow is an open-source platform for managing "
        "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
        "a company that specializes in big data and machine learning solutions. MLflow is "
        "designed to address the challenges that data scientists and machine learning "
        "engineers face when developing, training, and deploying machine learning models."
    },
)

custom_metric = make_genai_metric(
    name="correctness",
    version="v1",
    definition="Correctness refers to how well the generated output matches "
    "or aligns with the reference or ground truth text that is considered "
    "accurate and appropriate for the given input. The ground truth serves as "
    "a benchmark against which the provided output is compared to determine the "
    "level of accuracy and fidelity.",
    grading_prompt="Correctness: If the answer correctly answer the question, below are the "
    "details for different scores: "
    "- Score 0: the answer is completely incorrect, doesn’t mention anything about "
    "the question or is completely contrary to the correct answer. "
    "- Score 1: the answer provides some relevance to the question and answer one aspect "
    "of the question correctly. "
    "- Score 2: the answer mostly answer the question but is missing or hallucinating on one "
    "critical aspect. "
    "- Score 4: the answer correctly answer the question and not missing any major aspect",
    examples=[example],
    model="openai:/gpt-3.5-turbo-16k",
    variables=["ground_truth"],
    parameters={"temperature": 0.0},
    greater_is_better=True,
    aggregations=["mean", "variance", "p90"],
)

eval_df = pd.DataFrame(
    {
        "input": [
            "What is MLflow?",
            "What is Spark?",
            "What is Python?",
            "What is Delta Lake?",
            "What is Apache Iceberg?",
            "What is AWS?",
            "What is Kubernetes?",
            "What is Docker?",
            "What is Hadoop?",
            "What is TensorFlow?",
            "What is PyTorch?",
            "What is NoSQL?",
            "What is GraphQL?",
            "What is REST API?",
            "What is Git?",
            "What is CI/CD?",
        ],
        "prediction": [
            "MLflow is an open-source platform for managing machine learning workflows, including experiment tracking, model packaging, versioning, and deployment, simplifying the ML lifecycle.",
            "Apache Spark is an open-source, distributed computing system designed for big data processing and analytics. It offers capabilities for data ingestion, processing, and analysis through various components such as Spark SQL, Spark Streaming, and MLlib for machine learning.",
            "Python is a high-level, interpreted programming language known for its readability and versatility. It is widely used for web development, data analysis, artificial intelligence, and more.",
            "Delta Lake is a storage layer that brings reliability to Data Lakes. It works over your existing data and is fully compatible with API's like Spark, Hive, Presto, etc",
            "Apache Iceberg is an open-source table format for large, distributed datasets. It is designed to improve on the limitations of older formats like Parquet and Avro.",
            "AWS is Amazon Web Services, a cloud computing platform provided by Amazon. It offers a wide range of services such as computing power, storage, and databases to businesses and developers.",
            "Kubernetes is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications.",
            "Docker is a platform that enables developers to create, deploy, and run applications in containers, which are lightweight, standalone packages that contain all the dependencies needed to run the code.",
            "Hadoop is an open-source framework designed for distributed storage and processing of big data using the MapReduce programming model.",
            "TensorFlow is an open-source machine learning framework developed by Google that provides tools for training and deploying artificial intelligence models.",
            "PyTorch is an open-source machine learning library for Python that's widely used for deep learning and artificial intelligence applications.",
            "NoSQL is a type of database that's optimized for handling large volumes of unstructured data. It stands for 'not only SQL' and differs from traditional relational databases.",
            "GraphQL is a query language for APIs that enables clients to request exactly the data they need, making it more efficient than REST.",
            "REST API is an architectural style for designing networked applications. It stands for Representational State Transfer and is based on standard HTTP methods.",
            "Git is a distributed version control system used for tracking changes in source code during software development.",
            "CI/CD stands for Continuous Integration and Continuous Deployment. It's a set of practices for automating the testing and deployment of software.",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is designed to address the challenges that data scientists and machine learning engineers face when developing, training, and deploying machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data processing and analytics. It was developed in response to limitations of the Hadoop MapReduce computing model, offering improvements in speed and ease of use. Spark provides libraries for various tasks such as data ingestion, processing, and analysis through its components like Spark SQL for structured data, Spark Streaming for real-time data processing, and MLlib for machine learning tasks",
            "Python is a high-level programming language that was created by Guido van Rossum and released in 1991. It emphasizes code readability and allows developers to express concepts in fewer lines of code than languages like C++ or Java. Python is used in various domains, including web development, scientific computing, data analysis, and machine learning.",
            "Delta Lake is an open-source storage layer that enhances the reliability and performance of Data Lakes. It brings ACID transactions to these systems, ensuring data integrity for big data workloads. The layer is compatible with popular big data APIs such as Apache Spark, Hive, and Presto, enabling you to query your data without changing your existing architecture.",
            "Apache Iceberg is an open-source table format built for handling large, distributed datasets in a more efficient manner. It provides features like ACID transactions, versioning, and schema evolution to improve upon existing data storage formats like Parquet and Avro. Iceberg is widely used with data processing engines like Apache Spark, Trino, and Flink.",
            "Amazon Web Services (AWS) is a subsidiary of Amazon providing on-demand cloud computing platforms and APIs to individuals, companies, and governments. AWS offers a broad set of products that include computing power, database storage, content delivery, and other functionalities to help businesses scale and grow. These services are designed to work together to provide a comprehensive cloud computing environment.",
            "Kubernetes is an open-source container orchestration platform developed by Google. It provides automation for deploying, scaling, and managing containerized applications across clusters of servers. Kubernetes also provides features like load balancing, rollbacks, and self-healing capabilities.",
            "Docker is a containerization platform that packages an application and its dependencies into a container, which can be run consistently on any infrastructure. It was created to simplify the process of developing, shipping, and running applications irrespective of the environment.",
            "Apache Hadoop is an open-source, distributed computing framework that facilitates the processing and storage of large data sets across clusters of computers. It is based on the MapReduce programming model and consists of a distributed file system called Hadoop Distributed File System (HDFS).",
            "TensorFlow is an open-source machine learning library developed by the Google Brain team. It is designed to facilitate the development and deployment of machine learning models for various tasks such as natural language processing, computer vision, and recommendation systems.",
            "PyTorch is an open-source machine learning library developed primarily by Facebook's AI Research lab. It provides a dynamic computation graph, making it particularly useful for research and experimentation. It supports a wide array of machine learning algorithms, including neural networks and gradient-boosted trees.",
            "NoSQL databases are non-relational databases designed to scale horizontally and manage large volumes of structured and unstructured data. They can handle various data models like document, key-value, and graph. NoSQL databases are known for their ability to handle high-throughput applications.",
            "GraphQL is a query language developed by Facebook in 2015 for more efficient API calls. Unlike REST, which exposes multiple endpoints for different resources, GraphQL exposes a single endpoint for all interactions, enabling clients to request exactly the data they need.",
            "REST (Representational State Transfer) API is a set of constraints for building web services that allow for the manipulation of resources via stateless operations. RESTful APIs use standard HTTP methods like GET, POST, PUT, and DELETE for data exchange.",
            "Git is a distributed version control system created by Linus Torvalds in 2005. It is used to track changes in source code during software development, enabling multiple developers to work on a project simultaneously while maintaining a history of changes.",
            "CI/CD, or Continuous Integration and Continuous Deployment, is a set of DevOps practices that automate the software delivery process, from code changes in the repository to deployment to production. CI ensures code is integrated into a shared repository frequently, and CD automates the delivery of this code to production.",
        ],
    }
)

metric_value = custom_metric.eval_fn(eval_df)

# Print the metric value and its scores and justifications.
print(f"Metric scores: {metric_value.scores}")
print(f"Metric justifications: {metric_value.justifications}")

num_good_scores = sum([1 for x in metric_value.scores if x is not None])
num_good_justifications = sum([1 for x in metric_value.justifications if x is not None])

print(f"Percent good scores: {num_good_scores / len(metric_value.scores)}")
print(f"Percent good justifications: {num_good_justifications / len(metric_value.justifications)}")
