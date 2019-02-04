"""
Constants shared by several backend store implementations.
"""

# Path to default location for backend when using local FileStore or ArtifactStore.
# Also used as default location for artifacts, when not provided, in non local file based backends
# (eg MySQL)
DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH = "./mlruns"
