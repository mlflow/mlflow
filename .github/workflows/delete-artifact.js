/**
 * Main function to handle documentation preview comments
 * @param {object} params - Parameters object containing context and github
 * @param {object} params.github - GitHub API client
 * @param {object} params.context - GitHub context
 * @param {object} params.env - Environment variables
 */
module.exports = async ({ github, context, env }) => {
  const artifactName = env.ARTIFACT_NAME;
  const runId = env.RUN_ID;

  if (!artifactName || !runId) {
    throw new Error("Missing required parameters: ARTIFACT_NAME, RUN_ID");
  }

  const { owner, repo } = context.repo;

  try {
    // INFO: https://octokit.github.io/rest.js/v22/#actions-list-workflow-run-artifacts
    const {
      data: { artifacts },
    } = await github.rest.actions.listWorkflowRunArtifacts({
      owner,
      repo,
      run_id: runId,
      name: artifactName,
    });

    const [artifact] = artifacts;

    // INFO: https://octokit.github.io/rest.js/v22/#actions-delete-artifact
    await github.rest.actions.deleteArtifact({
      owner,
      repo,
      artifact_id: artifact.id,
    });
  } catch (error) {
    console.error(`Could not find or delete the artifact for ${runId} and ${artifactName}`);
    throw error;
  }
};
