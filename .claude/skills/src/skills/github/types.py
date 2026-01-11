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


class JobStep(BaseModel):
    name: str
    status: str
    conclusion: str | None
    number: int
    started_at: str | None
    completed_at: str | None


class Job(BaseModel):
    id: int
    run_id: int
    url: str  # API URL (e.g., "https://api.github.com/repos/owner/repo/actions/jobs/123")
    name: str
    workflow_name: str  # Name of the workflow
    status: str
    conclusion: str | None
    html_url: str
    started_at: str | None
    completed_at: str | None
    steps: list[JobStep] = []


class JobRun(BaseModel):
    id: int
    name: str
    head_sha: str
    status: str
    conclusion: str | None
    html_url: str
    created_at: str
    updated_at: str


# TODO: Add more models when needed (no need to add them in this task tho)
