import mlflow
import random
import requests
from PIL import Image
from io import BytesIO

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

nested_params = {
    'model': {
        'layer1': {
            'neurons': 128,
            'activation': 'relu'
        },
        'layer2': {
            'neurons': 64,
            'activation': 'sigmoid'
        }
    },
    'optimizer': {
        'type': 'adam',
        'learning_rate': 0.001
    }
}

# Set up MLFlow experiment
mlflow.set_experiment("Test Experiment")

# Function to generate random parameters and metrics
def generate_random_data(num_params, num_metrics):
    params = {f"param_{i}": random.random() for i in range(num_params)}
    metrics = {f"metric_{i}": random.random() * 100 for i in range(num_metrics)}
    return params, metrics

# Number of runs, parameters, and metrics
num_runs = 5
num_params = 200
num_metrics = 200

for run in range(num_runs):
    with mlflow.start_run():
        # Generate random parameters and metrics
        params, metrics = generate_random_data(num_params, num_metrics)
        
        # Log parameters
        mlflow.log_params(params)

        # Flatten parameters
        flat_params = flatten_dict(nested_params)

        # Log flattened parameters
        mlflow.log_params(flat_params)

        # Log nested params (as JSON string)
        mlflow.log_param("nested_params", nested_params)

        # Download the image
        image_url1 = "https://www.mlflow.org/docs/latest/_static/images/intro/prompt-engineering-ui.png"
        response1 = requests.get(image_url1)
        image1 = Image.open(BytesIO(response1.content))
        image_path1 = "prompt-engineering-ui.png"
        image1.save(image_path1)

        # Log image
        mlflow.log_artifact(image_path1, "images")

        
        # Download the image
        image_url2 = "https://getwallpapers.com/wallpaper/full/2/b/b/1199715-most-popular-4k-ultra-hd-nature-wallpaper-3840x2160-picture.jpg"
        response2 = requests.get(image_url2)
        image2 = Image.open(BytesIO(response2.content))
        image_path2 = "4k wallpaper.jpg"
        image2.save(image_path2)

        # Log image
        mlflow.log_artifact(image_path2, "images")

        
        # Download the image
        image_url3 = "https://wallpaperaccess.com/full/676960.jpg"
        response3 = requests.get(image_url3)
        image3 = Image.open(BytesIO(response3.content))
        image_path3 = "8k wallpaper.jpg"
        image3.save(image_path3)

        # Log image
        mlflow.log_artifact(image_path3, "images")

        image_url4 = "https://cdn.abcteach.com/abcteach-content-free/docs/free_preview/s/smallbird01lowres_p.png"
        response4 = requests.get(image_url4)
        image4 = Image.open(BytesIO(response4.content))
        image_path4 = "bird.png"
        image4.save(image_path4)

        # Log image
        mlflow.log_artifact(image_path4, "images")

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        print(f"Run {run+1}/{num_runs} logged successfully!")

print("All runs logged.")
