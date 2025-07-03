/**
 * Script to manage documentation preview comments on pull requests.
 */

const MARKER = "<!-- documentation preview -->";

/**
 * Create or update a PR comment with documentation preview information
 * @param {object} github - GitHub API client
 * @param {string} repo - Repository name in format "owner/repo"
 * @param {string} pullNumber - Pull request number
 * @param {string} commentBody - Comment body content
 */
async function upsertComment(github, repo, pullNumber, commentBody) {
  const [owner, repoName] = repo.split("/");

  // Get existing comments on the PR
  const { data: comments } = await github.rest.issues.listComments({
    owner,
    repo: repoName,
    issue_number: pullNumber,
    per_page: 100,
  });

  // Find existing preview docs comment
  const existingComment = comments.find((comment) => comment.body.includes(MARKER));
  const commentBodyWithMarker = `${MARKER}\n\n${commentBody}`;

  if (!existingComment) {
    console.log("Creating comment");
    await github.rest.issues.createComment({
      owner,
      repo: repoName,
      issue_number: pullNumber,
      body: commentBodyWithMarker,
    });
  } else {
    console.log("Updating comment");
    await github.rest.issues.updateComment({
      owner,
      repo: repoName,
      comment_id: existingComment.id,
      body: commentBodyWithMarker,
    });
  }
}

/**
 * Generate the comment template for documentation preview
 * @param {string} commitSha - Git commit SHA
 * @param {string} workflowRunLink - Link to the workflow run
 * @param {string} docsWorkflowRunUrl - Link to the docs workflow run
 * @param {string} mainMessage - Main message content
 * @returns {string} Comment template
 */
function getCommentTemplate(commitSha, workflowRunLink, docsWorkflowRunUrl, mainMessage) {
  return `
Documentation preview for ${commitSha} ${mainMessage}

<details>
<summary>More info</summary>

- Ignore this comment if this PR does not change the documentation.
- The preview is updated when a new commit is pushed to this PR.
- This comment was created by [this workflow run](${workflowRunLink}).
- The documentation was built by [this workflow run](${docsWorkflowRunUrl}).

</details>
`;
}

/**
 * Main function to handle documentation preview comments
 * @param {object} params - Parameters object containing context and github
 * @param {object} params.github - GitHub API client
 * @param {object} params.context - GitHub context
 * @param {object} params.env - Environment variables
 */
module.exports = async ({ github, context, env }) => {
  const commitSha = env.COMMIT_SHA;
  const pullNumber = env.PULL_NUMBER;
  const workflowRunId = env.WORKFLOW_RUN_ID;
  const stage = env.STAGE;
  const netlifyUrl = env.NETLIFY_URL;
  const docsWorkflowRunUrl = env.DOCS_WORKFLOW_RUN_URL;

  // Validate required parameters
  if (!commitSha || !pullNumber || !workflowRunId || !stage || !docsWorkflowRunUrl) {
    throw new Error(
      "Missing required parameters: commit-sha, pull-number, workflow-run-id, stage, docs-workflow-run-url"
    );
  }

  if (!["completed", "failed"].includes(stage)) {
    throw new Error("Stage must be either 'completed' or 'failed'");
  }

  if (stage === "completed" && !netlifyUrl) {
    throw new Error("netlify-url is required for completed stage");
  }

  const { owner, repo } = context.repo;
  const workflowRunLink = `https://github.com/${owner}/${repo}/actions/runs/${workflowRunId}`;

  let mainMessage;
  if (stage === "completed") {
    mainMessage = `is available at:\n\n- ${netlifyUrl}`;
  } else if (stage === "failed") {
    mainMessage = "failed to build or deploy.";
  }

  const commentBody = getCommentTemplate(
    commitSha,
    workflowRunLink,
    docsWorkflowRunUrl,
    mainMessage
  );
  await upsertComment(github, `${owner}/${repo}`, pullNumber, commentBody);
};
