import type { getOctokit } from "@actions/github";
import type { context as ContextType } from "@actions/github";
import type { components } from "@octokit/openapi-webhooks-types";

type GitHub = ReturnType<typeof getOctokit>;
type Context = typeof ContextType;
type WorkflowDispatch = components["schemas"]["webhook-workflow-dispatch"];
type ReleaseEvent = { release: { tag_name: string } };

interface ReleaseInfo {
  releaseVersion: string;
  releaseTag: string;
  releaseLabel: string;
  nextPatchLabel: string;
  releaseBranch: string;
}

interface CommitInfo {
  commit: {
    message: string;
  };
}

/**
 * Extract release information from either release event or workflow_dispatch input
 */
function extractReleaseInfo(context: Context): ReleaseInfo {
  let releaseVersion: string;
  let releaseTag: string;

  if (context.eventName === "workflow_dispatch") {
    // Manual trigger with version parameter
    const payload = context.payload as WorkflowDispatch;
    releaseVersion = payload.inputs?.release_version as string;
    if (!releaseVersion) {
      throw new Error("release_version input is required for workflow_dispatch");
    }
    releaseTag = releaseVersion.startsWith("v") ? releaseVersion : `v${releaseVersion}`;
    releaseVersion = releaseVersion.replace(/^v/, ""); // Remove 'v' prefix if present
    console.log(`Processing manual workflow for release: ${releaseTag} (${releaseVersion})`);
  } else {
    // Automatic trigger from release event
    const payload = context.payload as ReleaseEvent;
    const release = payload.release;
    if (!release) {
      throw new Error("Release information not found in payload");
    }
    releaseTag = release.tag_name;
    releaseVersion = releaseTag.replace(/^v/, ""); // Remove 'v' prefix if present
    console.log(`Processing release event: ${releaseTag} (${releaseVersion})`);
  }

  const versionMatch = releaseVersion.match(/^(\d+)\.(\d+)\.(\d+)$/);
  if (!versionMatch) {
    console.log(`Skipping invalid release: ${releaseVersion}`);
    throw new Error(`Invalid version format: ${releaseVersion}`);
  }

  const [, major, minor, patch] = versionMatch;
  const nextPatchVersion = `${major}.${minor}.${parseInt(patch) + 1}`;
  const releaseLabel = `v${releaseVersion}`;
  const nextPatchLabel = `v${nextPatchVersion}`;

  // Get release branch name (e.g., branch-3.1 for v3.1.4)
  const releaseBranch = `branch-${major}.${minor}`;

  console.log(`Release label: ${releaseLabel}`);
  console.log(`Next patch label: ${nextPatchLabel}`);
  console.log(`Release branch: ${releaseBranch}`);

  return {
    releaseVersion,
    releaseTag,
    releaseLabel,
    nextPatchLabel,
    releaseBranch,
  };
}

/**
 * Helper function to extract PR number from commit message
 */
function extractPRNumberFromCommitMessage(commitMessage: string): number | null {
  const prRegex = /\(#(\d+)\)$/;
  const lines = commitMessage.split("\n");

  for (const line of lines) {
    const match = line.trim().match(prRegex);
    if (match) {
      return parseInt(match[1], 10);
    }
  }

  return null;
}

/**
 * Extract PR numbers from release branch commits
 */
async function extractPRNumbersFromBranch(
  github: GitHub,
  context: Context,
  releaseBranch: string
): Promise<Set<number>> {
  const releasePRNumbers = new Set<number>();

  try {
    const commits: CommitInfo[] = await github.paginate(github.rest.repos.listCommits, {
      owner: context.repo.owner,
      repo: context.repo.repo,
      sha: releaseBranch,
      since: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(), // Last 30 days
    });

    for (const commit of commits) {
      const prNumber = extractPRNumberFromCommitMessage(commit.commit.message);
      if (prNumber) {
        releasePRNumbers.add(prNumber);
      }
    }

    console.log(`Found ${releasePRNumbers.size} PR numbers from ${releaseBranch} commits`);
  } catch (error) {
    if (
      error instanceof Error &&
      "status" in error &&
      (error as { status: number }).status === 404
    ) {
      console.log(
        `Release branch '${releaseBranch}' not found. This may be expected for new releases.`
      );
      console.log("Skipping commit analysis - will update all PRs with the release label.");
    } else {
      throw error;
    }
  }

  return releasePRNumbers;
}

/**
 * Fetch all merged PRs with a specific label
 */
async function fetchPRsWithLabel(
  github: GitHub,
  context: Context,
  releaseLabel: string
): Promise<Array<{ number: number; pull_request?: any; state: string }>> {
  const allIssues = await github.paginate(github.rest.issues.listForRepo, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    labels: releaseLabel,
    state: "all",
  });

  const prsWithReleaseLabel = allIssues.filter((item) => {
    if (!item.pull_request) return false;
    if (item.state === "open") return true;
    if (item.state === "closed" && item.pull_request.merged_at) return true;
    return false;
  });

  console.log(`Found ${prsWithReleaseLabel.length} PRs with label ${releaseLabel}`);
  return prsWithReleaseLabel;
}

/**
 * Update PR labels for PRs not included in release
 */
async function updatePRLabels(
  github: GitHub,
  context: Context,
  prsWithReleaseLabel: Array<{ number: number; pull_request?: any; state: string }>,
  releasePRNumbers: Set<number>,
  releaseLabel: string,
  nextPatchLabel: string
): Promise<void> {
  const pullRequests = prsWithReleaseLabel.filter((item) => item.pull_request);
  console.log(
    `Processing ${pullRequests.length} PRs (filtered out ${
      prsWithReleaseLabel.length - pullRequests.length
    } issues)`
  );

  const prsToUpdate: number[] = [];

  for (const pr of pullRequests) {
    if (releasePRNumbers.has(pr.number)) continue;
    prsToUpdate.push(pr.number);
  }

  console.log(`Found ${prsToUpdate.length} PRs that need label updates: ${prsToUpdate.join(", ")}`);

  for (const prNumber of prsToUpdate) {
    try {
      await github.rest.issues.removeLabel({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: prNumber,
        name: releaseLabel,
      });

      await github.rest.issues.addLabels({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: prNumber,
        labels: [nextPatchLabel],
      });

      console.log(`Updated PR #${prNumber}: ${releaseLabel} → ${nextPatchLabel}`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.log(`Warning: Failed to update labels for PR #${prNumber}: ${errorMessage}`);
    }
  }
}

/**
 * Main function to update release labels
 *
 * This script checks all PRs labeled with a release version and updates
 * their labels to the next patch version if they weren't actually included
 * in the release (handles cherry-picked commits properly).
 */
export async function updateReleaseLabels({
  github,
  context,
}: {
  github: GitHub;
  context: Context;
}): Promise<void> {
  try {
    const releaseInfo = extractReleaseInfo(context);

    const releasePRNumbers = await extractPRNumbersFromBranch(
      github,
      context,
      releaseInfo.releaseBranch
    );

    const prsWithReleaseLabel = await fetchPRsWithLabel(github, context, releaseInfo.releaseLabel);

    await updatePRLabels(
      github,
      context,
      prsWithReleaseLabel,
      releasePRNumbers,
      releaseInfo.releaseLabel,
      releaseInfo.nextPatchLabel
    );
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`Error updating release labels: ${errorMessage}`);
    throw error;
  }
}
