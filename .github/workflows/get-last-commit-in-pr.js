module.exports = async ({ github, core, context }) => {
  const { number: pull_number } = context.payload.pull_request;

  const { data } = await github.pulls.get({
    owner: "mlflow",
    repo: "mlflow",
    pull_number,
  });

  console.log(data.head.sha);
  console.log(context.payload.after);

  return data.head.sha === context.payload.after;
};
