const fs = require("fs");
const path = require("path");

/**
 * Upload artifacts to a GitHub release
 * @param {Object} params
 * @param {Object} params.github - GitHub API client
 * @param {Object} params.context - GitHub context
 * @param {string} params.artifactDir - Path to the directory containing artifacts
 */
module.exports = async ({ github, context, artifactDir }) => {
  if (!fs.existsSync(artifactDir)) {
    throw new Error(`Artifacts directory not found: ${artifactDir}`);
  }

  const artifactFiles = fs
    .readdirSync(artifactDir)
    .filter((f) => fs.statSync(path.join(artifactDir, f)).isFile());

  if (artifactFiles.length === 0) {
    throw new Error(`No artifacts found in ${artifactDir}`);
  }

  // First, try to get existing release
  const { owner, repo } = context.repo;
  let release;
  let releaseExists = false;
  try {
    const { data } = await github.rest.repos.getReleaseByTag({
      owner,
      repo,
      tag: "nightly",
    });
    release = data;
    releaseExists = true;
    console.log(`Found existing release: ${release.id}`);
  } catch (error) {
    if (error.status !== 404) {
      throw error;
    }
  }

  if (releaseExists) {
    // Update existing release
    console.log("Updating existing nightly release...");
    const { data: updatedRelease } = await github.rest.repos.updateRelease({
      owner,
      repo,
      release_id: release.id,
      tag_name: "nightly",
      target_commitish: context.sha,
      name: `Nightly Build ${new Date().toISOString().split("T")[0]}`,
      body: `This is an automated nightly build of MLflow.\n\n**Last updated:** ${new Date().toUTCString()}\n\n**Note:** This release is automatically updated daily with the latest changes from the master branch.`,
      prerelease: true,
      make_latest: "false",
    });
    release = updatedRelease;
    console.log(`Updated existing release: ${release.id}`);

    // Delete existing assets
    console.log("Fetching existing assets...");
    const { data: assets } = await github.rest.repos.listReleaseAssets({
      owner,
      repo,
      release_id: release.id,
      per_page: 100,
    });

    for (const asset of assets) {
      console.log(`Deleting old asset: ${asset.name}`);
      await github.rest.repos.deleteReleaseAsset({
        owner,
        repo,
        asset_id: asset.id,
      });
    }
  } else {
    // Create a new release (this will also create the tag)
    console.log("Creating new nightly release...");
    const { data: newRelease } = await github.rest.repos.createRelease({
      owner,
      repo,
      tag_name: "nightly",
      target_commitish: context.sha,
      name: `Nightly Build ${new Date().toISOString().split("T")[0]}`,
      body: `This is an automated nightly build of MLflow.\n\n**Last updated:** ${new Date().toUTCString()}\n\n**Note:** This release is automatically updated daily with the latest changes from the master branch.`,
      prerelease: true,
      make_latest: "false",
    });
    release = newRelease;
    console.log(`Created new release: ${release.id}`);
  }

  // Upload all artifacts
  for (const artifactName of artifactFiles) {
    const artifactPath = path.join(artifactDir, artifactName);
    const contentType = getContentType(artifactName);

    console.log(`Uploading ${artifactName}...`);
    const artifactData = fs.readFileSync(artifactPath);
    await github.rest.repos.uploadReleaseAsset({
      owner,
      repo,
      release_id: release.id,
      name: artifactName,
      data: artifactData,
      headers: {
        "content-type": contentType,
        "content-length": artifactData.length,
      },
    });

    console.log(`Successfully uploaded ${artifactName}`);
  }

  console.log("All artifacts uploaded successfully");
};

/**
 * Get content type based on file extension
 * @param {string} filename
 * @returns {string} Content type
 */
function getContentType(filename) {
  const ext = path.extname(filename).toLowerCase();
  switch (ext) {
    case ".whl":
    case ".zip":
      return "application/zip";
    case ".tar.gz":
      return "application/gzip";
    case ".jar":
      return "application/java-archive";
    default:
      return "application/octet-stream";
  }
}
