module.exports = ({ core, context }) => {
  const { body, user, html_url } = context.payload.pull_request;

  // Skip validation on pull requests created by the automation bot
  if (user.login === "mlflow-automation") {
    console.log(
      "Skipping validation since this pull request was created by the automation bot"
    );
    return;
  }

  const categories = [
    "rn/feature",
    "rn/breaking-change",
    "rn/bug-fix",
    "rn/documentation",
    "rn/none",
  ];

  // Extract selected release-note categories from the PR description
  const regexp = /^- \[(?<selected>.*?)\] ?`(?<category>.+?)`/gm;
  const selectedCategories = [];
  let match = regexp.exec(body);
  while (match != null) {
    const selected = match.groups.selected.trim().toLocaleLowerCase() === "x";
    const category = match.groups.category.trim();

    if (categories.includes(category) && selected) {
      selectedCategories.push(category);
    }
    match = regexp.exec(body);
  }

  console.log("Selected release-note categories:");
  console.log(selectedCategories);

  // ".github/pull_request_template.md" must contain an HTML anchor with this name
  const anchorName = "release-note-category";

  // GitHub prefixes anchor names in markdown with "user-content-"
  const anchorUrl = `${html_url}#user-content-${anchorName}`;

  // If no release-note categories is selected, set the action status to "failed"
  const numSelected = selectedCategories.length;
  if (numSelected === 0) {
    core.setFailed(
      "At least one release-note category must be selected: " + anchorUrl
    );
  }
};
