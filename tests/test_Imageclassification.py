
import mlflow
import os
from mlflow import pyfunc
from datasets import load_dataset
from transformers import ViTImageProcessor,ViTForImageClassification
def vision_model_save():
    Model_name="facebook/deit-base-patch16-224"
    model_path= "facebook_deit_base_patch16_224"
    if not (os.path.exists(model_path)):
       image_processor = ViTImageProcessor.from_pretrained(Model_name)
       model = ViTForImageClassification.from_pretrained(Model_name)
       components={"model": model,"image_processor":image_processor}
       mlflow.transformers.save_model(
                transformers_model=components,
                path=model_path,
        )
    return  model_path   
def load_model():
    modelpath=vision_model_save()
    model = mlflow.pyfunc.load_model(modelpath)
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    result=model.predict(url)
    
    return result

