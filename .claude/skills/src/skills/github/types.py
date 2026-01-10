from pydantic import BaseModel


class GitRef(BaseModel):
    sha: str
    ref: str


class PullRequest(BaseModel):
    title: str
    body: str | None
    head: GitRef


# TODO: Add more models when needed (no need to add them in this task tho)
