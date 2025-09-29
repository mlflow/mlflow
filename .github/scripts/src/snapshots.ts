import { readFileSync, existsSync, readdirSync, statSync } from "fs";
import { join, basename } from "path";
import type { getOctokit } from "@actions/github";
import type { context as ContextType } from "@actions/github";

type GitHub = ReturnType<typeof getOctokit>;
type Context = typeof ContextType;
type GitHubAsset = Awaited<
  ReturnType<GitHub["rest"]["repos"]["listReleaseAssets"]>
>["data"][number];
type GitHubRelease = Awaited<ReturnType<GitHub["rest"]["repos"]["getReleaseByTag"]>>["data"];

// Constants
const RELEASE_TAG = "nightly";
const DAYS_TO_KEEP = 3;

interface SnapshotParams {
  github: GitHub;
  context: Context;
  artifactDir: string;
}

/**
 * Check if artifact file type is supported
 */
function isSupportedArtifact(filename: string): boolean {
  return /\.(whl|jar|tar\.gz)$/.test(filename);
}

/**
 * Get content type based on file extension
 */
function getContentType(filename: string): string {
  if (filename.match(/\.whl$/)) {
    return "application/zip";
  } else if (filename.match(/\.tar\.gz$/)) {
    return "application/gzip";
  } else if (filename.match(/\.jar$/)) {
    return "application/java-archive";
  }
  throw new Error(
    `Unsupported file type for content type: ${filename}. Only .whl, .jar, and .tar.gz are supported.`
  );
}

/**
 * Check if asset should be deleted based on age
 */
function shouldDeleteAsset(asset: GitHubAsset, daysToKeep: number): boolean {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - daysToKeep);
  const assetDate = new Date(asset.created_at);
  return assetDate < cutoffDate;
}

/**
 * Add commit SHA to filename based on file type
 */
function addShaToFilename(filename: string, sha: string): string {
  const shortSha = sha.substring(0, 7);

  // Match .whl files
  if (filename.match(/\.whl$/)) {
    // For wheel files, insert SHA as build tag before .whl extension
    // Build tags must start with a digit, so prefix with "0"
    const wheelParts = filename.match(/^(.+?)(-py[^-]+(?:-[^-]+)*\.whl)$/);
    if (wheelParts) {
      return `${wheelParts[1]}-0${shortSha}${wheelParts[2]}`;
    }
    // Fallback for non-standard wheel names
    return filename.replace(/\.whl$/, `-0${shortSha}.whl`);
  }

  // Match .jar files
  if (filename.match(/\.jar$/)) {
    return filename.replace(/\.jar$/, `-${shortSha}.jar`);
  }

  // Match .tar.gz files
  if (filename.match(/\.tar\.gz$/)) {
    return filename.replace(/\.tar\.gz$/, `-${shortSha}.tar.gz`);
  }

  throw new Error(
    `Unexpected file extension for: ${filename}. Only .whl, .jar, and .tar.gz are supported.`
  );
}

/**
 * Upload artifacts to a GitHub release
 */
export async function uploadSnapshots({
  github,
  context,
  artifactDir,
}: SnapshotParams): Promise<void> {
  if (!existsSync(artifactDir)) {
    throw new Error(`Artifacts directory not found: ${artifactDir}`);
  }

  const artifactFiles = readdirSync(artifactDir)
    .map((f) => join(artifactDir, f))
    .filter((f) => statSync(f).isFile());

  if (artifactFiles.length === 0) {
    throw new Error(`No artifacts found in ${artifactDir}`);
  }

  // Check for unsupported file types
  const unsupportedFiles = artifactFiles.filter((f) => !isSupportedArtifact(f));
  if (unsupportedFiles.length > 0) {
    const names = unsupportedFiles.map((f) => `  - ${basename(f)}`).join("\n");
    throw new Error(
      `Found unsupported file types:\n${names}\nOnly .whl, .jar, and .tar.gz files are supported.`
    );
  }

  // Check if the release already exists
  const { owner, repo } = context.repo;
  let release: GitHubRelease;
  let releaseExists = false;
  try {
    const { data } = await github.rest.repos.getReleaseByTag({
      owner,
      repo,
      tag: RELEASE_TAG,
    });
    release = data;
    releaseExists = true;
    console.log(`Found existing release: ${release.id}`);
  } catch (error: any) {
    if (error.status !== 404) {
      throw error;
    }
  }

  const releaseParams = {
    owner,
    repo,
    tag_name: RELEASE_TAG,
    target_commitish: context.sha,
    name: `Nightly Build ${new Date().toISOString().split("T")[0]}`,
    body: `This is an automated nightly build of MLflow.

**Last updated:** ${new Date().toUTCString()}
**Commit:** ${context.sha}

**Note:** This release is automatically updated daily with the latest changes from the master branch.`,
    prerelease: true,
    make_latest: "false" as const,
  };

  if (releaseExists) {
    console.log("Updating existing nightly release...");
    const { data: updatedRelease } = await github.rest.repos.updateRelease({
      ...releaseParams,
      release_id: release!.id,
    });
    release = updatedRelease;
    console.log(`Updated existing release: ${release.id}`);
  } else {
    console.log("Creating new nightly release...");
    const { data: newRelease } = await github.rest.repos.createRelease(releaseParams);
    release = newRelease;
    console.log(`Created new release: ${release.id}`);
  }

  console.log("Fetching all existing assets...");
  const allAssets: GitHubAsset[] = await github.paginate(github.rest.repos.listReleaseAssets, {
    owner,
    repo,
    release_id: release.id,
  });
  console.log(`Found ${allAssets.length} existing assets`);

  // Delete old assets.
  for (const asset of allAssets) {
    if (shouldDeleteAsset(asset, DAYS_TO_KEEP)) {
      const assetDate = new Date(asset.created_at).toISOString().split("T")[0];
      console.log(`Deleting old asset (created ${assetDate}): ${asset.name}`);
      await github.rest.repos.deleteReleaseAsset({
        owner,
        repo,
        asset_id: asset.id,
      });
    }
  }

  // Filter to get remaining assets after deletion
  const remainingAssets = allAssets.filter((asset) => !shouldDeleteAsset(asset, DAYS_TO_KEEP));

  // Upload all artifacts
  for (const artifactPath of artifactFiles) {
    const artifactName = basename(artifactPath);
    const contentType = getContentType(artifactName);
    const nameWithSha = addShaToFilename(artifactName, context.sha);

    // Check if artifact with SHA already exists in remaining assets
    if (remainingAssets.some((asset) => asset.name === nameWithSha)) {
      console.log(`Artifact already exists: ${nameWithSha} (skipping upload)`);
      continue;
    }

    console.log(`Uploading ${artifactName} as ${nameWithSha}...`);
    const artifactData = readFileSync(artifactPath);
    await github.rest.repos.uploadReleaseAsset({
      owner,
      repo,
      release_id: release.id,
      name: nameWithSha,
      data: artifactData as unknown as string,
      headers: {
        "content-type": contentType,
        "content-length": artifactData.length,
      },
    });

    console.log(`Successfully uploaded ${artifactName} as ${nameWithSha}`);
  }

  console.log("All artifacts uploaded successfully");
}
