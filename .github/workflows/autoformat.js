const createCommitStatus = async (context, github, sha, state) => {
  const { workflow, runId } = context;
  const { owner, repo } = context.repo;
  const target_url = `https://github.com/${owner}/${repo}/actions/runs/${runId}`;
  await github.repos.createCommitStatus({
    owner,
    repo,
    sha,
    state,
    target_url,
    description: sha,
    context: workflow,
  });
};

const createStatus = async (context, github) => {
  const { owner, repo } = context.repo;
  const pull_number = context.issue.number;
  const pr = await github.pulls.get({ owner, repo, pull_number });
  const { sha, ref } = pr.data.head;
  await createCommitStatus(context, github, sha, 'pending');
  return {
    repository: `${owner}/${repo}`,
    pull_number,
    sha,
    ref,
  };
};

const updateStatus = async (context, github, sha, needs) => {
  const failed = Object.values(needs).some(({ result }) => result === 'failure');
  const state = failed ? 'failure' : 'success';
  await createCommitStatus(context, github, sha, state);
};

module.exports = {
  createStatus,
  updateStatus,
};
