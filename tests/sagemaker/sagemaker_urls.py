from tests.sagemaker.sagemaker_service_mock import SageMakerResponse

url_bases = [
    "https?://sagemaker.(.+).amazonaws.com",
]

url_paths = {
    '{0}/$': SageMakerResponse.dispatch,
}
