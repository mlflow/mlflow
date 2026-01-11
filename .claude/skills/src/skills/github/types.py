from pydantic import BaseModel


class GitRef(BaseModel):
    sha: str
    ref: str


class PullRequest(BaseModel):
    title: str
    body: str | None
    head: GitRef


class ReviewComment(BaseModel):
    id: int
    body: str
    author: str
    createdAt: str


class ReviewThread(BaseModel):
    thread_id: str
    line: int | None
    startLine: int | None
    diffHunk: str | None
    comments: list[ReviewComment]


# TODO: Add more models when needed (no need to add them in this task tho)
