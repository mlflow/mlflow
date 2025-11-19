const fs = require("fs");

async function download({ github, context }) {
  const { owner, repo } = context.repo;
  const run_id = context.payload.workflow_run.id;

  const { data } = await github.rest.actions.listWorkflowRunArtifacts({
    owner,
    repo,
    run_id,
  });
  const artifact = data.artifacts.find((a) => a.name === "assign_reviewer_payload");
  if (!artifact) {
    throw new Error("assign_reviewer_payload artifact not found");
  }

  const download = await github.rest.actions.downloadArtifact({
    owner,
    repo,
    artifact_id: artifact.id,
    archive_format: "zip",
  });

  fs.writeFileSync(`${process.env.GITHUB_WORKSPACE}/assign_payload.zip`, Buffer.from(download.data));
}

async function assign({ github, context }) {
  const payloadPath = `${process.env.GITHUB_WORKSPACE}/payload.json`;
  const content = fs.readFileSync(payloadPath, "utf8");
  const payload = JSON.parse(content);

  const {
    pr_number,
    reviewer,
  } = payload;

  const { owner, repo } = context.repo;

  let permission = "none";
  try {
    const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
      owner,
      repo,
      username: reviewer,
    });
    permission = data.permission;
  } catch (err) {
    console.log(`Could not fetch permission for ${reviewer}: ${err.message}`);
    return;
  }

  if (!["write", "maintain", "admin"].includes(permission)) {
    console.log(`User ${reviewer} has permission '${permission}'; skipping assignment.`);
    return;
  }

  try {
    await github.rest.issues.addAssignees({
      owner,
      repo,
      issue_number: pr_number,
      assignees: [reviewer],
    });
    console.log(`Assigned ${reviewer} to PR #${pr_number}.`);
  } catch (err) {
    console.log(`Failed to assign ${reviewer}: ${err.message}`);
  }
}

module.exports = { download, assign };

