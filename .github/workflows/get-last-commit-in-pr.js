module.exports = ({ github, core, context }) => {
  const { number: pull_number } = context.payload.pull_request;

  const { data } = await github.rest.pulls.get({
    owner: "mlflow",
    repo: "mlflow",
    pull_number,
  });

  console.log(data);
};
