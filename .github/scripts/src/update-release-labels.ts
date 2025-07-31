import type { getOctokit } from "@actions/github";
import type { context as ContextType } from "@actions/github";

type GitHub = ReturnType<typeof getOctokit>;
type Context = typeof ContextType;

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
    releaseVersion = (context.payload.inputs as any)?.release_version;
    if (!releaseVersion) {
      throw new Error("release_version input is required for workflow_dispatch");
    }
    releaseTag = releaseVersion.startsWith("v") ? releaseVersion : `v${releaseVersion}`;
    releaseVersion = releaseVersion.replace(/^v/, ""); // Remove 'v' prefix if present
    console.log(`Processing manual workflow for release: ${releaseTag} (${releaseVersion})`);
  } else {
    // Automatic trigger from release event
    const release = (context.payload as any)?.release;
    if (!release) {
      throw new Error("Release information not found in payload");
    }
    releaseTag = release.tag_name;
    releaseVersion = releaseTag.replace(/^v/, ""); // Remove 'v' prefix if present
    console.log(`Processing release event: ${releaseTag} (${releaseVersion})`);
  }

  const versionMatch = releaseVersion.match(/^(\d+)\.(\d+)\.(\d+)$/);
  if (!versionMatch) {
    console.log(`Skipping unofficial release: ${releaseVersion}`);
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
  const prRegex = /\(#(\d+)\)/;
  const lines = commitMessage.split("\n");
  
  for (const line of lines) {
    const match = line.match(prRegex);
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
): Promise<{ releasePRNumbers: Set<number>; totalCommits: number }> {
  const releasePRNumbers = new Set<number>();
  let totalCommits = 0;

  try {
    const commits: CommitInfo[] = await github.paginate(github.rest.repos.listCommits, {
      owner: context.repo.owner,
      repo: context.repo.repo,
      sha: releaseBranch,
      since: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(), // Last 30 days
    });

    totalCommits = commits.length;

    for (const commit of commits) {
      const prNumber = extractPRNumberFromCommitMessage(commit.commit.message);
      if (prNumber) {
        releasePRNumbers.add(prNumber);
      }
    }
  } catch (error: any) {
    if (error.status === 404) {
      console.log(`Release branch '${releaseBranch}' not found. This may be expected for new releases.`);
      console.log("Skipping commit analysis - will update all PRs with the release label.");
    } else {
      throw error; // Re-throw other errors
    }
  }

  console.log(`Found ${totalCommits} commits in ${releaseBranch} with ${releasePRNumbers.size} PR numbers`);
  return { releasePRNumbers, totalCommits };
}

/**
 * Fetch all PRs with a specific label
 */
async function fetchPRsWithLabel(
  github: GitHub,
  context: Context,
  releaseLabel: string
): Promise<Array<{ number: number; pull_request?: any; title?: string }>> {
  // Find all PRs with the release label using github.paginate
  const prsWithReleaseLabel = await github.paginate(github.rest.issues.listForRepo, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    labels: releaseLabel,
    state: "all",
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
  prsWithReleaseLabel: Array<{ number: number; pull_request?: any; title?: string }>,
  releasePRNumbers: Set<number>,
  releaseLabel: string,
  nextPatchLabel: string,
  releaseVersion: string
): Promise<number> {
  let updatedPRs = 0;
  
  const pullRequests = prsWithReleaseLabel.filter(item => item.pull_request);
  console.log(`Processing ${pullRequests.length} PRs (filtered out ${prsWithReleaseLabel.length - pullRequests.length} issues)`);

  const prsToUpdate: number[] = [];

  for (const pr of pullRequests) {
    const prIncludedInRelease = releasePRNumbers.has(pr.number);
    if (prIncludedInRelease) continue;
    
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

      updatedPRs++;
      console.log(`Updated PR #${prNumber}: ${releaseLabel} → ${nextPatchLabel}`);
    } catch (error: any) {
      console.log(`Warning: Failed to update labels for PR #${prNumber}: ${error.message}`);
    }
  }

  return updatedPRs;
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

    const { releasePRNumbers } = await extractPRNumbersFromBranch(
      github,
      context,
      releaseInfo.releaseBranch
    );

    const prsWithReleaseLabel = await fetchPRsWithLabel(github, context, releaseInfo.releaseLabel);

    const updatedPRs = await updatePRLabels(
      github,
      context,
      prsWithReleaseLabel,
      releasePRNumbers,
      releaseInfo.releaseLabel,
      releaseInfo.nextPatchLabel,
      releaseInfo.releaseVersion
    );

    console.log(`Release label update completed. Updated ${updatedPRs} PRs.`);
  } catch (error: any) {
    console.error(`Error updating release labels: ${error.message}`);
    throw error;
  }
}

export default updateReleaseLabels;