# Development

## Build devcontainer image locally

```bash
# This command may take a while to complete.
DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain docker build -f .devcontainer/Dockerfile.devcontainer -t mlflow-devcontainer .

# Create a container using the built image and run a command in it
docker run --rm -it -v $(pwd):/workspaces/mlflow mlflow-devcontainer [COMMAND]
```

## Build devcontainer using locally-built devcontainer image

Once the devcontainer image is built, modify `.devcontainer/docker-compose.yml` as follows and build the devcontainer:

```diff
 version: '3'
 services:
   mlflow:
-    image: 'ghcr.io/mlflow/mlflow-devcontainer'
+    image: 'mlflow-devcontainer'
```

## Push devcontainer image

[The devcontainer workflow](../.github/workflows/devcontainer.yml) builds the devcontainer image and pushes it to GitHub Container Registry as `ghcr.io/mlflow/mlflow-devcontainer`.

## References

- [`devcontainer.json` reference](https://code.visualstudio.com/docs/remote/devcontainerjson-reference)
- [Working with the Container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
