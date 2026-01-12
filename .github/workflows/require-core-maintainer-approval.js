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

const EXEMPTION_RULES = [
  // Exemption for GenAI evaluation PRs.
  {
    authors: ["alkispoly-db", "AveshCSingh", "danielseong1", "smoorjani", "SomtochiUmeh", "xsh310"],
    allowedPatterns: [
      /^mlflow\/genai\//,
      /^tests\/genai\//,
      /^docs\//,
      /^mlflow\/entities\/(assessment|dataset|evaluation|scorer)/,
    ],
    excludedPatterns: [/^mlflow\/genai\/(agent_server|git_versioning|prompts|optimize)\//],
  },
  // Exemption for UI PRs.
  {
    authors: ["daniellok-db", "danielseong1", "hubertzub-db"],
    allowedPatterns: [/^mlflow\/server\/js\//],
  },
];

function matchesAnyPattern(path, patterns) {
  if (!patterns) {
    return false;
  }
  return patterns.some((pattern) => pattern.test(path));
}

function isAllowedPath(path, rule) {
  return (
    matchesAnyPattern(path, rule.allowedPatterns) && !matchesAnyPattern(path, rule.excludedPatterns)
  );
}

function isExempted(authorLogin, files) {
  let filesToCheck = files;
  for (const rule of EXEMPTION_RULES) {
    if (rule.authors.includes(authorLogin)) {
      filesToCheck = filesToCheck.filter(
        ({ filename, previous_filename }) =>
          // Keep files where NOT all before/after file paths are allowed by the rule.
          ![filename, previous_filename].filter(Boolean).every((path) => isAllowedPath(path, rule))
      );
      if (filesToCheck.length === 0) {
        return true;
      }
    }
  }
  return false;
}

function hasAnyApproval(reviews) {
  return reviews.some(({ state }) => state === "APPROVED");
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
