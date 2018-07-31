import sys
import requests
import json
import pandas as pd
import numpy as np
import argparse

from datetime import datetime
from collections import namedtuple

Query = namedtuple('Query', ['url', 'headers', 'data'])

class BenchmarkResults:

    def __init__(self, num_queries, thru, mean_lat, p99_lat, lat_std, all_lats):
        self.num_queries = num_queries
        self.thru = thru
        self.mean_lat = mean_lat
        self.p99_lat = p99_lat
        self.lat_std = lat_std
        self.all_lats = all_lats

    def save(self, path):
        results_dict = self.__dict__
        with open(path, "w") as f:
            json.dump(results_dict, f, indent=4)

def construct_mleap_query(url, headers, json_data, idx):
    new_data = dict(json_data)
    first_row = str(new_data["rows"][0][0])
    first_words = first_row.split(" ")
    np.random.shuffle(first_words)
    new_first = " ".join(first_words) + " {}".format(idx)
    new_data["rows"][0][0] = new_first 
    new_query = Query(url=url, headers=headers, data=json.dumps(new_data))
    return new_query

def construct_sparkml_query(url, headers, json_data, idx):
    new_data = dict(json_data)
    text = str(new_data["text"])
    words = text.split(" ")
    np.random.shuffle(words)
    words.append(str(idx))
    new_data["text"] = " ".join(words)
    new_query = Query(url=url, headers=headers, data=json.dumps([new_data]))
    return new_query

def construct_queries(data_path, num_queries):
    url = "http://localhost:8080/invocations"
    headers = {"Content-type" : "application/json"}

    with open(data_path, "r") as f:
        json_data = json.load(f)

    queries = []
    for i in range(num_queries):
        if "text" in json_data.keys():
            new_query = construct_sparkml_query(url, headers, json_data, i)
        else:
            new_query = construct_mleap_query(url, headers, json_data, i)
        queries.append(new_query)

    return queries

def send_query(query):
    response = requests.post(query.url, headers=query.headers, data=query.data)
    print(response.text)
    return response.text

def benchmark(data_path, num_queries):
    print("Benchmarking {} queries".format(num_queries))

    queries = construct_queries(data_path, num_queries)
    begin = datetime.now()
    latencies = []
    for query in queries:
        before = datetime.now()
        send_query(query)
        after = datetime.now()
        latency = (after - before).total_seconds()
        latencies.append(latency)

    end = datetime.now()
    total_latency = (end - begin).total_seconds()
    thru = float(num_queries) / total_latency
    mean_lat = np.mean(latencies)
    p99_lat = np.percentile(latencies, 99)
    lat_std = np.std(latencies)

    return BenchmarkResults(num_queries=num_queries,
                            thru=thru, 
                            mean_lat=mean_lat, 
                            p99_lat=p99_lat, 
                            lat_std=lat_std,
                            all_lats=latencies)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark SparkML vs MLeap')
    parser.add_argument('--data_path', '-d', type=str, help="Path to query data")
    parser.add_argument('--output_path', '-o', type=str, help="Path to which to save results")
    parser.add_argument('--num_queries', '-n', type=int, default=1000, help="Number of queries to send during benchmark")

    args = parser.parse_args()
    
    results = benchmark(args.data_path, args.num_queries)
    results.save(args.output_path)

    print("Wrote results to: {}".format(args.output_path))
