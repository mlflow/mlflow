const { owner, repo } = context.repo;
const artifactsResp = await github.actions.listArtifactsForRepo({
  owner,
  repo,
});
const wheels = artifactsResp.data.artifacts.filter(a => a.endsWith(".whl"));

// The storage usage limit for a free github account is 500 MB. See the page below for details:
// https://docs.github.com/en/github/setting-up-and-managing-billing-and-payments-on-github/about-billing-for-github-actions
MAX_SIZE_IN_BYTES = 3_000_000; // 300 MB

let index = 0;
let sum = 0;
for (const { size_in_bytes } of wheels) {
  index = idx;
  sum += size_in_bytes;
  if (sum > MAX_SIZE_IN_BYTES) {
    break;
  }
}

if (sum <= MAX_SIZE_IN_BYTES) {
  return;
}

// Delete old wheels
const promises = wheels.slice(index).map(({ id: artifact_id }) =>
  github.actions.deleteArtifact({
    owner,
    repo,
    artifact_id,
  })
);
Promise.all(promises).then(results => console.log(results));
