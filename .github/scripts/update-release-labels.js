/**
 * GitHub Action script to update PR labels when releases are cut.
 *
 * This script checks all PRs labeled with a release version and updates
 * their labels to the next patch version if they weren't actually included
 * in the release (handles cherry-picked commits properly).
 */

module.exports = async ({ github, context }) => {
  // Get the release information
  const release = context.payload.release;
  const releaseTag = release.tag_name;
  const releaseVersion = releaseTag.replace(/^v/, ""); // Remove 'v' prefix if present

  console.log(`Processing release: ${releaseTag} (${releaseVersion})`);

  // Parse version components
  const versionMatch = releaseVersion.match(/^(\d+)\.(\d+)\.(\d+)$/);
  if (!versionMatch) {
    console.log(`Skipping unofficial release: ${releaseVersion}`);
    return;
  }

  const [, major, minor, patch] = versionMatch;
  const nextPatchVersion = `${major}.${minor}.${parseInt(patch) + 1}`;
  const releaseLabel = `v${releaseVersion}`;
  const nextPatchLabel = `v${nextPatchVersion}`;

  console.log(`Release label: ${releaseLabel}`);
  console.log(`Next patch label: ${nextPatchLabel}`);

  // Get release branch name (e.g., branch-3.1 for v3.1.4)
  const majorMinorVersion = releaseVersion.split(".").slice(0, 2).join(".");
  const releaseBranch = `branch-${majorMinorVersion}`;
  console.log(`Release branch: ${releaseBranch}`);

  // Get PR numbers from the release branch commits (similar to check_patch_prs.py)
  let releasePRNumbers = new Set();

  // Get commits from the release branch with pagination
  let page = 1;
  let hasMore = true;
  let totalCommits = 0;

  while (hasMore) {
    const branchCommits = await github.rest.repos.listCommits({
      owner: context.repo.owner,
      repo: context.repo.repo,
      sha: releaseBranch,
      per_page: 100,
      page: page,
      since: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString(), // Last 3 months
    });

    totalCommits += branchCommits.data.length;
    hasMore = branchCommits.data.length === 100;
    page++;

    // Extract PR numbers from commit messages using regex (same as check_patch_prs.py)
    const prRegex = /\(#(\d+)\)/;

    for (const commit of branchCommits.data) {
      const lines = commit.commit.message.split("\n");
      for (const line of lines) {
        const match = line.match(prRegex);
        if (match) {
          const prNumber = parseInt(match[1]);
          releasePRNumbers.add(prNumber);
          console.log(`Found PR #${prNumber} in release branch: ${line.trim()}`);
          break; // Stop after finding the first PR number in the commit
        }
      }
    }
  }

  console.log(`Found ${totalCommits} total commits in ${releaseBranch}`);
  console.log(`Extracted ${releasePRNumbers.size} PR numbers from release branch commits`);

  // Find all PRs with the release label with pagination
  let prsWithReleaseLabel = [];
  let prPageNum = 1;
  let hasMorePRs = true;

  while (hasMorePRs) {
    const prPageData = await github.rest.issues.listForRepo({
      owner: context.repo.owner,
      repo: context.repo.repo,
      labels: releaseLabel,
      state: "all",
      per_page: 100,
      page: prPageNum,
    });

    prsWithReleaseLabel.push(...prPageData.data);
    hasMorePRs = prPageData.data.length === 100;
    prPageNum++;
  }

  console.log(`Found ${prsWithReleaseLabel.length} PRs with label ${releaseLabel}`);
  console.log(`Found released PR numbers:`, Array.from(releasePRNumbers));

  // Check each PR to see if it was actually included in the release
  for (const pr of prsWithReleaseLabel) {
    if (!pr.pull_request) continue; // Skip issues

    console.log(`Checking PR #${pr.number}: ${pr.title}`);

    // Check if this PR number is in the release branch commits
    const prIncludedInRelease = releasePRNumbers.has(pr.number);
    console.log(`PR #${pr.number} included in release: ${prIncludedInRelease}`);

    if (!prIncludedInRelease) {
      console.log(`PR #${pr.number} was not included in release ${releaseVersion}, updating label`);

      // Remove the old release label
      try {
        await github.rest.issues.removeLabel({
          owner: context.repo.owner,
          repo: context.repo.repo,
          issue_number: pr.number,
          name: releaseLabel,
        });
        console.log(`Removed label ${releaseLabel} from PR #${pr.number}`);
      } catch (error) {
        console.log(`Error removing label from PR #${pr.number}: ${error.message}`);
      }

      // Add the next patch version label
      try {
        await github.rest.issues.addLabels({
          owner: context.repo.owner,
          repo: context.repo.repo,
          issue_number: pr.number,
          labels: [nextPatchLabel],
        });
        console.log(`Added label ${nextPatchLabel} to PR #${pr.number}`);
      } catch (error) {
        console.log(`Error adding label to PR #${pr.number}: ${error.message}`);
      }
    } else {
      console.log(`PR #${pr.number} was correctly included in release ${releaseVersion}`);
    }
  }

  console.log("Release label update process completed");
};
