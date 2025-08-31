/**
 * Script to manage documentation preview comments on pull requests.
 */

const path = require("path");

const MARKER = "<!-- documentation preview -->";

/**
 * Fetch changed files from a pull request
 * @param {object} params - Parameters object
 * @param {object} params.github - GitHub API client
 * @param {string} params.owner - Repository owner
 * @param {string} params.repo - Repository name
 * @param {string} params.pullNumber - Pull request number
 * @returns {Promise<string[]>} Array of changed file paths
 */
async function fetchChangedFiles({ github, owner, repo, pullNumber }) {
  const iterator = github.paginate.iterator(github.rest.pulls.listFiles, {
    owner,
    repo,
    pull_number: pullNumber,
    per_page: 100,
  });

  const changedFiles = [];
  for await (const { data } of iterator) {
    changedFiles.push(...data.map(({ filename }) => filename));
  }

  return changedFiles;
}

/**
 * Get changed documentation pages from the list of changed files
 * @param {string[]} changedFiles - Array of changed file paths
 * @returns {string[]} Array of documentation page paths
 */
function getChangedDocPages(changedFiles) {
  const DOCS_DIR = "docs/docs/";
  const changedPages = [];

  for (const file of changedFiles) {
    const ext = path.extname(file);
    if (ext !== ".md" && ext !== ".mdx") continue;
    if (!file.startsWith(DOCS_DIR)) continue;

    const relativePath = path.relative(DOCS_DIR, file);
    const { dir, name, base } = path.parse(relativePath);

    let pagePath;
    if (base === "index.mdx") {
      pagePath = dir;
    } else {
      pagePath = path.join(dir, name);
    }

    // Adjust classic-ml/ to ml/
    pagePath = pagePath.replace(/^classic-ml/, "ml");

    // Ensure forward slashes for web paths
    pagePath = pagePath.split(path.sep).join("/");

    changedPages.push(pagePath);
  }

  return changedPages;
}

/**
 * Create or update a PR comment with documentation preview information
 * @param {object} params - Parameters object
 * @param {object} params.github - GitHub API client
 * @param {string} params.owner - Repository owner
 * @param {string} params.repo - Repository name
 * @param {string} params.pullNumber - Pull request number
 * @param {string} params.commentBody - Comment body content
 */
async function upsertComment({ github, owner, repo, pullNumber, commentBody }) {
  // Get existing comments on the PR
  const { data: comments } = await github.rest.issues.listComments({
    owner,
    repo,
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
      repo,
      issue_number: pullNumber,
      body: commentBodyWithMarker,
    });
  } else {
    console.log("Updating comment");
    await github.rest.issues.updateComment({
      owner,
      repo,
      comment_id: existingComment.id,
      body: commentBodyWithMarker,
    });
  }
}

/**
 * Generate the comment template for documentation preview
 * @param {object} params - Parameters object
 * @param {string} params.commitSha - Git commit SHA
 * @param {string} params.workflowRunLink - Link to the workflow run
 * @param {string} params.docsWorkflowRunUrl - Link to the docs workflow run
 * @param {string} params.mainMessage - Main message content
 * @param {string[]} params.changedPages - Array of changed documentation page links
 * @returns {string} Comment template
 */
function getCommentTemplate({
  commitSha,
  workflowRunLink,
  docsWorkflowRunUrl,
  mainMessage,
  changedPages,
}) {
  let changedPagesSection = "";

  if (changedPages && changedPages.length > 0) {
    const pageLinks = changedPages.map((page) => `- ${page}`).join("\n");
    changedPagesSection = `

<details>
<summary>Changed Pages (${changedPages.length})</summary>

${pageLinks}

</details>
`;
  }

  return `
Documentation preview for ${commitSha} ${mainMessage}
${changedPagesSection}
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
  let changedPages = [];

  if (stage === "completed") {
    mainMessage = `is available at:\n\n- ${netlifyUrl}`;

    // Fetch changed files and get documentation pages
    try {
      const changedFiles = await fetchChangedFiles({ github, owner, repo, pullNumber });
      const docPages = getChangedDocPages(changedFiles);

      // Convert to clickable links if we have changed pages
      if (docPages.length > 0) {
        changedPages = docPages.map((page) => `[${page}](${netlifyUrl}/${page})`);
      }
    } catch (error) {
      console.error("Error fetching changed files:", error);
      // Continue without changed pages list
    }
  } else if (stage === "failed") {
    mainMessage = "failed to build or deploy.";
  }

  const commentBody = getCommentTemplate({
    commitSha,
    workflowRunLink,
    docsWorkflowRunUrl,
    mainMessage,
    changedPages,
  });
  await upsertComment({ github, owner, repo, pullNumber, commentBody });
};
