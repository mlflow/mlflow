/**
 * Script to manage documentation preview comments on pull requests.
 *
 * This script creates or updates a comment on a PR with links to the documentation
 * preview deployed on Netlify. It handles both successful deployments and failures,
 * providing appropriate feedback to PR authors and reviewers.
 */

const path = require("path");

const MARKER = "<!-- documentation preview -->";

/**
 * Fetch changed files from a pull request using GitHub API pagination.
 *
 * @param {object} params - Parameters object
 * @param {object} params.github - GitHub API client from actions/github-script
 * @param {string} params.owner - Repository owner (organization or user)
 * @param {string} params.repo - Repository name
 * @param {string} params.pullNumber - Pull request number as string
 * @returns {Promise<string[]>} Array of changed file paths relative to repository root
 * @throws {Error} When GitHub API request fails
 */
async function fetchChangedFiles({ github, owner, repo, pullNumber }) {
  try {
    const iterator = github.paginate.iterator(github.rest.pulls.listFiles, {
      owner,
      repo,
      pull_number: parseInt(pullNumber, 10),
      per_page: 100,
    });

    const changedFiles = [];
    for await (const { data } of iterator) {
      changedFiles.push(...data.map(({ filename }) => filename));
    }

    return changedFiles;
  } catch (error) {
    console.error(`Failed to fetch changed files for PR #${pullNumber}:`, error.message);
    throw error;
  }
}

/**
 * Extract documentation page paths from changed files.
 * Only processes .md and .mdx files within the docs/docs/ directory.
 *
 * @param {string[]} changedFiles - Array of changed file paths relative to repository root
 * @returns {string[]} Array of documentation page paths for linking (without file extensions)
 */
function getChangedDocPages(changedFiles) {
  const DOCS_DIR = "docs/docs/";
  const changedPages = [];

  for (const file of changedFiles) {
    const ext = path.extname(file);

    // Only process markdown files
    if (ext !== ".md" && ext !== ".mdx") continue;

    // Only process files in the docs directory
    if (!file.startsWith(DOCS_DIR)) continue;

    const relativePath = path.relative(DOCS_DIR, file);
    const { dir, name, base } = path.parse(relativePath);

    let pagePath;
    // Handle index files - they represent the directory itself
    if (base === "index.mdx") {
      pagePath = dir;
    } else {
      pagePath = path.join(dir, name);
    }

    // Handle special case: classic-ml/ maps to ml/ in the deployed docs
    pagePath = pagePath.replace(/^classic-ml/, "ml");

    // Normalize path separators for web URLs
    pagePath = pagePath.split(path.sep).join("/");

    changedPages.push(pagePath);
  }

  return changedPages;
}

/**
 * Create or update a PR comment with documentation preview information.
 * Uses a marker to identify existing preview comments for updates.
 *
 * @param {object} params - Parameters object
 * @param {object} params.github - GitHub API client from actions/github-script
 * @param {string} params.owner - Repository owner (organization or user)
 * @param {string} params.repo - Repository name
 * @param {string} params.pullNumber - Pull request number as string
 * @param {string} params.commentBody - Comment body content (without marker)
 * @throws {Error} When GitHub API request fails
 */
async function upsertComment({ github, owner, repo, pullNumber, commentBody }) {
  try {
    // Fetch existing comments on the PR
    const { data: comments } = await github.rest.issues.listComments({
      owner,
      repo,
      issue_number: parseInt(pullNumber, 10),
      per_page: 100,
    });

    // Find existing preview docs comment using our marker
    const existingComment = comments.find((comment) => comment.body.includes(MARKER));
    const commentBodyWithMarker = `${MARKER}\n\n${commentBody}`;

    if (!existingComment) {
      console.log(`Creating new preview comment for PR #${pullNumber}`);
      await github.rest.issues.createComment({
        owner,
        repo,
        issue_number: parseInt(pullNumber, 10),
        body: commentBodyWithMarker,
      });
    } else {
      console.log(`Updating existing preview comment for PR #${pullNumber}`);
      await github.rest.issues.updateComment({
        owner,
        repo,
        comment_id: existingComment.id,
        body: commentBodyWithMarker,
      });
    }
  } catch (error) {
    console.error(`Failed to upsert comment for PR #${pullNumber}:`, error.message);
    throw error;
  }
}

/**
 * Generate the markdown comment body for documentation preview.
 *
 * @param {object} params - Parameters object
 * @param {string} params.commitSha - Git commit SHA (first 7 characters used for display)
 * @param {string} params.workflowRunLink - URL to the current workflow run
 * @param {string} params.docsWorkflowRunUrl - URL to the docs build workflow run
 * @param {string} params.mainMessage - Main status message (e.g., deployment URL or failure notice)
 * @param {string[]} [params.changedPages] - Array of changed documentation page links
 * @returns {string} Formatted markdown comment body
 */
function getCommentTemplate({
  commitSha,
  workflowRunLink,
  docsWorkflowRunUrl,
  mainMessage,
  changedPages = [],
}) {
  const shortSha = commitSha.substring(0, 7);
  let changedPagesSection = "";

  if (changedPages.length > 0) {
    const pageLinks = changedPages.map((page) => `- ${page}`).join("\n");
    changedPagesSection = `

<details>
<summary>Changed Pages (${changedPages.length})</summary>

${pageLinks}

</details>
`;
  }

  return `
Documentation preview for \`${shortSha}\` ${mainMessage}
${changedPagesSection}
<details>
<summary>More info</summary>

- Ignore this comment if this PR does not change the documentation.
- The preview is updated when a new commit is pushed to this PR.
- This comment was created by [this workflow run](${workflowRunLink}).
- The documentation was built by [this workflow run](${docsWorkflowRunUrl}).

</details>
`.trim();
}

/**
 * Main function to handle documentation preview comments.
 * This function is called by GitHub Actions and orchestrates the entire process
 * of creating or updating documentation preview comments on pull requests.
 *
 * @param {object} params - Parameters object from GitHub Actions
 * @param {object} params.github - GitHub API client from actions/github-script
 * @param {object} params.context - GitHub Actions context object
 * @param {object} params.env - Environment variables from workflow
 * @param {string} params.env.COMMIT_SHA - Git commit SHA
 * @param {string} params.env.PULL_NUMBER - Pull request number
 * @param {string} params.env.WORKFLOW_RUN_ID - Current workflow run ID
 * @param {string} params.env.STAGE - Stage: 'completed' for success, 'failed' for failure
 * @param {string} [params.env.NETLIFY_URL] - Netlify deployment URL (required for 'completed' stage)
 * @param {string} params.env.DOCS_WORKFLOW_RUN_URL - URL to the docs build workflow run
 * @throws {Error} When required parameters are missing or invalid
 */
module.exports = async ({ github, context, env }) => {
  // Extract and validate required environment variables
  const {
    COMMIT_SHA: commitSha,
    PULL_NUMBER: pullNumber,
    WORKFLOW_RUN_ID: workflowRunId,
    STAGE: stage,
    NETLIFY_URL: netlifyUrl,
    DOCS_WORKFLOW_RUN_URL: docsWorkflowRunUrl,
  } = env;

  // Validate required parameters
  if (!commitSha || !pullNumber || !workflowRunId || !stage || !docsWorkflowRunUrl) {
    const missing = [
      !commitSha && "COMMIT_SHA",
      !pullNumber && "PULL_NUMBER",
      !workflowRunId && "WORKFLOW_RUN_ID",
      !stage && "STAGE",
      !docsWorkflowRunUrl && "DOCS_WORKFLOW_RUN_URL",
    ].filter(Boolean);

    throw new Error(`Missing required environment variables: ${missing.join(", ")}`);
  }

  // Validate stage parameter
  const validStages = ["completed", "failed"];
  if (!validStages.includes(stage)) {
    throw new Error(`Stage must be one of: ${validStages.join(", ")}. Got: ${stage}`);
  }

  // Validate netlify URL for completed stage
  if (stage === "completed" && !netlifyUrl) {
    throw new Error("NETLIFY_URL is required when stage is 'completed'");
  }

  const { owner, repo } = context.repo;
  const workflowRunLink = `https://github.com/${owner}/${repo}/actions/runs/${workflowRunId}`;

  let mainMessage;
  let changedPages = [];

  try {
    if (stage === "completed") {
      mainMessage = `is available at:\n\n- ${netlifyUrl}`;

      // Fetch changed files and generate documentation page links
      try {
        const changedFiles = await fetchChangedFiles({ github, owner, repo, pullNumber });
        const docPages = getChangedDocPages(changedFiles);

        // Convert to clickable links if we have changed pages
        if (docPages.length > 0) {
          changedPages = docPages.map((page) => `[${page}](${netlifyUrl}/${page})`);
        }
      } catch (error) {
        console.warn(
          "Could not fetch changed files, continuing without changed pages list:",
          error.message
        );
        // Continue without changed pages list - this is not critical to the core functionality
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
    console.log(
      `Successfully handled documentation preview comment for PR #${pullNumber} (stage: ${stage})`
    );
  } catch (error) {
    console.error(
      `Failed to handle documentation preview comment for PR #${pullNumber}:`,
      error.message
    );
    throw error;
  }
};
