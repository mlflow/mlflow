const fs = require("fs");
const path = require("path");

/**
 * Upload artifacts to a GitHub release
 * @param {Object} params
 * @param {Object} params.github - GitHub API client
 * @param {Object} params.context - GitHub context
 * @param {string|string[]} params.artifacts - Path(s) to artifact file(s)
 */
module.exports = async ({ github, context, artifacts }) => {
  const { owner, repo } = context.repo;

  // Normalize to array
  const artifactPaths = Array.isArray(artifacts) ? artifacts : [artifacts];

  // Validate all artifacts exist
  for (const artifactPath of artifactPaths) {
    if (!fs.existsSync(artifactPath)) {
      throw new Error(`Artifact not found: ${artifactPath}`);
    }
  }

  // First, try to get existing release
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

  // If release exists, delete it (this also deletes the tag)
  if (releaseExists) {
    console.log("Deleting existing nightly release and tag...");
    await github.rest.repos.deleteRelease({
      owner,
      repo,
      release_id: release.id,
    });
  }

  // Create a new release (this will also create the tag)
  console.log("Creating new nightly release...");
  const { data: newRelease } = await github.rest.repos.createRelease({
    owner,
    repo,
    tag_name: "nightly",
    target_commitish: context.sha, // Use the current commit SHA
    name: `Nightly Build ${new Date().toISOString().split("T")[0]}`,
    body: `This is an automated nightly build of MLflow.\n\n**Last updated:** ${new Date().toUTCString()}\n\n**Note:** This release is automatically updated daily with the latest changes from the master branch.`,
    prerelease: true,
    make_latest: "false",
  });
  release = newRelease;
  console.log(`Created new release: ${release.id}`);

  // Upload all artifacts
  for (const artifactPath of artifactPaths) {
    const artifactName = path.basename(artifactPath);
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
    case ".jar":
      return "application/java-archive";
    default:
      return "application/octet-stream";
  }
}
