async function getMaintainers({ github, context }) {
  const collaborators = await github.paginate(github.rest.repos.listCollaborators, {
    owner: context.repo.owner,
    repo: context.repo.repo,
  });
  return collaborators
    .filter(({ role_name }) => ["admin", "maintain"].includes(role_name))
    .map(({ login }) => login)
    .sort();
}

const EXEMPTIONS = [
  // Exemption for GenAI evaluation PRs.
  {
    "authors": [
      "alkispoly-db",
      "AveshCSingh",
      "danielseong1",
      "smoorjani",
      "SomtochiUmeh",
      "xsh310",
    ],
    "allowed": [
      "mlflow/genai/**",
      "tests/genai/**",
      "docs/**",
    ],
    "excludes": [
      "mlflow/genai/prompts/**",
      "mlflow/genai/optimize/**",
    ],
  }
]


function isAllowedPath(path) {
  return EXEMPTIONS.some(({ allowed, excludes }) => {
    return allowed.some((allowed) => path.startsWith(allowed)) && !excludes.some((exclude) => path.startsWith(exclude));
  });
}

function isExempted(authorLogin, files) {
  return EXEMPTIONS.some(({ authors }) => authors.includes(authorLogin)) || files.every(({ filename, previous_filename }) =>
    [filename, previous_filename].filter(Boolean).every(isAllowedPath));
}

module.exports = async ({ github, context, core }) => {
  const maintainers = await getMaintainers({ github, context });
  const reviews = await github.paginate(github.rest.pulls.listReviews, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    pull_number: context.issue.number,
  });
  const maintainerApproved = reviews.some(
    ({ state, user: { login } }) => state === "APPROVED" && maintainers.includes(login)
  );

  const { pull_request: pr } = context.payload;
  const authorLogin = pr?.user?.login;

  const files = await github.paginate(github.rest.pulls.listFiles, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    pull_number: context.issue.number,
  });

  if (isExempted(authorLogin, files)) {
    if (!hasAnyApproval(reviews)) {
      core.setFailed(
        "PR from exempted author needs at least one approval (maintainer approval not required)."
      );
    }
    return;
  }

  if (!maintainerApproved) {
    const maintainerList = maintainers.join(", ");
    const message = `This PR requires an approval from at least one of the core maintainers: ${maintainerList}.`;
    core.setFailed(message);
  }
};
