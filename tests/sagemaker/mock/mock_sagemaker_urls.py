from tests.sagemaker.mock import SageMakerResponse

url_bases = [
    "https?://api.sagemaker.(.+).amazonaws.com",
]

url_paths = {
    "{0}/$": SageMakerResponse.dispatch,
}
