from pathlib import Path

import yaml

_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "push-images.yml"


def test_floating_image_tags_are_promoted_after_versioned_images_are_published():
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    jobs = workflow["jobs"]
    build_steps = jobs["push-images"]["steps"]

    metadata_steps = {
        step["id"]: step
        for step in build_steps
        if step.get("uses", "").startswith("docker/metadata-action@")
    }
    assert metadata_steps.keys() == {"meta", "meta-full", "modelmeta"}
    for step in metadata_steps.values():
        assert step["with"]["flavor"] == "latest=false"
        assert "type=raw,value=latest" not in step["with"]["tags"]

    validation_index = next(
        index for index, step in enumerate(build_steps) if step.get("name") == "Validate image tags"
    )
    first_push_index = next(
        index
        for index, step in enumerate(build_steps)
        if step.get("with", {}).get("push") is True or "docker push" in step.get("run", "")
    )
    metadata_indexes = [build_steps.index(step) for step in metadata_steps.values()]
    assert max(metadata_indexes) < validation_index < first_push_index

    promotion_job = jobs["promote-latest"]
    assert promotion_job["needs"] == "push-images"
    assert promotion_job["permissions"] == {"contents": "read", "packages": "write"}
    assert promotion_job["concurrency"] == {
        "group": "push-images-promote-latest",
        "cancel-in-progress": False,
    }
    latest_check_script = next(
        step["run"]
        for step in promotion_job["steps"]
        if step.get("name") == "Check GitHub's latest release"
    )
    api_failure_branch = latest_check_script.split("elif", maxsplit=1)[0]
    assert "exit 1" in api_failure_branch

    promotion_script = next(
        step["run"]
        for step in promotion_job["steps"]
        if step.get("name") == "Promote floating image tags"
    )
    latest_check = 'gh api "repos/$GITHUB_REPOSITORY/releases/latest" --jq .tag_name'
    first_write = "docker buildx imagetools create --prefer-index=false"
    assert promotion_script.count(latest_check) == 1
    assert promotion_script.index(latest_check) < promotion_script.index(first_write)
    assert promotion_script.count(first_write) == 3

    expected_promotions = {
        (
            "ghcr.io/mlflow/mlflow:latest",
            '"ghcr.io/mlflow/mlflow:$RELEASE_TAG"',
        ),
        (
            "ghcr.io/mlflow/mlflow:latest-full",
            '"ghcr.io/mlflow/mlflow:$RELEASE_TAG-full"',
        ),
        (
            "ghcr.io/mlflow/model-server:latest",
            '"ghcr.io/mlflow/model-server:$RELEASE_TAG"',
        ),
    }
    for target, source in expected_promotions:
        command = "\n".join((
            f"{first_write} \\",
            f"  --tag {target} \\",
            f"  {source}",
        ))
        assert command in promotion_script
