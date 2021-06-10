module.exports = async ({ github, core, context }) => {
  const { sha } = context;
  const { number: pull_number } = context.payload.pull_request;

  const { data } = await github.pulls.get({
    owner: "mlflow",
    repo: "mlflow",
    pull_number,
  });

  const headSha = data.head.sha;

  console.log(data);
  console.log(context);
  console.log(sha);
  console.log(headSha);
};
