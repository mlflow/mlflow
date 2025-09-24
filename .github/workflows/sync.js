const { execSync } = require("child_process");

const shouldSync = (comment) => {
  return comment.body.trim() === "/sync";
};

const getPullInfo = async (context, github) => {
  const { owner, repo } = context.repo;
  const pull_number = context.issue.number;
  const pr = await github.rest.pulls.get({ owner, repo, pull_number });
  const {
    sha: head_sha,
    ref: head_ref,
    repo: { full_name },
  } = pr.data.head;
  const { sha: base_sha, ref: base_ref, repo: base_repo } = pr.data.base;
  return {
    repository: full_name,
    pull_number,
    head_sha,
    head_ref,
    base_sha,
    base_ref,
    base_repo: base_repo.full_name,
    author_association: pr.data.author_association,
    maintainer_can_modify: pr.data.maintainer_can_modify,
    mergeable: pr.data.mergeable,
    mergeable_state: pr.data.mergeable_state,
  };
};

const createInitialReaction = async (context, github) => {
  const { owner, repo } = context.repo;
  const { id: comment_id } = context.payload.comment;
  const { runId } = context;

  // Add rocket reaction
  await github.rest.reactions.createForIssueComment({
    owner,
    repo,
    comment_id,
    content: "rocket",
  });

  // Add workflow run link with pending emoji
  const workflowRunUrl = `https://github.com/${owner}/${repo}/actions/runs/${runId}`;
  const body = `⏳ [Sync workflow started](${workflowRunUrl})`;

  const response = await github.rest.issues.createComment({
    owner,
    repo,
    issue_number: context.issue.number,
    body,
  });

  return response.data.id;
};

const isAuthorAllowed = ({ author_association, user }) => {
  return (
    ["owner", "member", "collaborator"].includes(author_association.toLowerCase()) ||
    // Allow Copilot and mlflow-app bot to run this workflow
    (user &&
      user.type.toLowerCase() === "bot" &&
      ["copilot", "mlflow-app[bot]"].includes(user.login.toLowerCase()))
  );
};

const validateAuthorPermissions = async (context, github, core) => {
  const { comment } = context.payload;
  const { owner, repo } = context.repo;
  const { runId } = context;

  if (
    !isAuthorAllowed({
      author_association: comment.author_association,
      user: comment.user,
    })
  ) {
    const workflowRunUrl = `https://github.com/${owner}/${repo}/actions/runs/${runId}`;
    const message = `❌ **Sync failed**: Only repository owners, members, or collaborators can use the /sync command. @${comment.user.login} (${comment.author_association}) does not have sufficient permissions.

**Details:** [View workflow run](${workflowRunUrl})`;

    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: context.issue.number,
      body: message,
    });

    core.setFailed(`User ${comment.user.login} does not have sufficient permissions`);
  }
};

const validatePRConditions = async (context, github, pullInfo) => {
  const { owner, repo } = context.repo;
  const { runId } = context;
  const workflowRunUrl = `https://github.com/${owner}/${repo}/actions/runs/${runId}`;

  // Check if it's a fork PR
  const isForkPR = pullInfo.repository !== pullInfo.base_repo;

  // For fork PRs, check if maintainers can modify
  if (isForkPR && !pullInfo.maintainer_can_modify) {
    const message = `❌ **Sync failed**: For fork PRs, the "Allow edits and access to secrets by maintainers" checkbox must be checked.

Please:
1. Check the "Allow edits and access to secrets by maintainers" checkbox on this pull request
2. Comment \`/sync\` again

**Details:** [View workflow run](${workflowRunUrl})`;

    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: context.issue.number,
      body: message,
    });

    throw new Error("Fork PR does not allow maintainer edits");
  }

  // Check for merge conflicts
  if (pullInfo.mergeable === false) {
    const message = `❌ **Sync failed**: This PR has merge conflicts that must be resolved before syncing.

Please resolve the conflicts and then comment \`/sync\` again.

**Details:** [View workflow run](${workflowRunUrl})`;

    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: context.issue.number,
      body: message,
    });

    throw new Error("PR has merge conflicts");
  }

  // Check if mergeable state is unknown (GitHub is still computing)
  if (pullInfo.mergeable === null) {
    // Wait a bit and check again
    await new Promise((resolve) => setTimeout(resolve, 5000));

    const updatedPR = await github.rest.pulls.get({
      owner,
      repo,
      pull_number: pullInfo.pull_number,
    });

    if (updatedPR.data.mergeable === false) {
      const message = `❌ **Sync failed**: This PR has merge conflicts that must be resolved before syncing.

Please resolve the conflicts and then comment \`/sync\` again.

**Details:** [View workflow run](${workflowRunUrl})`;

      await github.rest.issues.createComment({
        owner,
        repo,
        issue_number: context.issue.number,
        body: message,
      });

      throw new Error("PR has merge conflicts");
    }
  }
};

const performSync = async (context, github, pullInfo, botToken) => {
  const { owner, repo } = context.repo;

  try {
    // Configure git
    execSync('git config user.name "mlflow-app[bot]"');
    execSync('git config user.email "mlflow-app[bot]@users.noreply.github.com"');

    // We're already on the PR head branch, just need to fetch and merge base
    const baseRepoUrl = `https://github.com/${pullInfo.base_repo}.git`;

    console.log(`Setting up base remote and fetching base branch`);

    // Clean up any existing remote
    try {
      execSync("git remote remove base", { stdio: "ignore" });
    } catch (e) {
      /* ignore */
    }

    execSync(`git remote add base ${baseRepoUrl}`);
    execSync(`git fetch base ${pullInfo.base_ref}`);

    console.log(`Merging base branch: ${pullInfo.base_ref}`);
    execSync(`git merge base/${pullInfo.base_ref} --no-edit`);

    console.log(`Pushing updated branch`);
    execSync(`git push origin HEAD`);

    console.log("Sync completed successfully");
  } catch (error) {
    console.error("Sync failed:", error.message);

    const { runId } = context;
    const workflowRunUrl = `https://github.com/${owner}/${repo}/actions/runs/${runId}`;
    const message = `❌ **Sync failed**: An error occurred during the sync process.

Error: \`${error.message}\`

**Details:** [View workflow run](${workflowRunUrl})`;

    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: context.issue.number,
      body: message,
    });

    throw error;
  }
};

const updateCommentStatus = async (context, github, success, workflowCommentId) => {
  const { owner, repo } = context.repo;
  const { runId } = context;
  const workflowRunUrl = `https://github.com/${owner}/${repo}/actions/runs/${runId}`;

  if (workflowCommentId) {
    const emoji = success ? "✅" : "❌";
    const status = success ? "completed successfully" : "failed";
    const updatedBody = `${emoji} [Sync workflow ${status}](${workflowRunUrl})`;

    try {
      await github.rest.issues.updateComment({
        owner,
        repo,
        comment_id: workflowCommentId,
        body: updatedBody,
      });
    } catch (error) {
      console.error("Failed to update workflow comment:", error.message);
    }
  }

  // If successful, add a success message
  if (success) {
    const message = `✅ **Sync completed**: The PR branch has been successfully updated with the latest changes from the base branch.

**Details:** [View workflow run](${workflowRunUrl})`;

    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: context.issue.number,
      body: message,
    });
  }
};

module.exports = {
  shouldSync,
  getPullInfo,
  createInitialReaction,
  validateAuthorPermissions,
  validatePRConditions,
  performSync,
  updateCommentStatus,
};
