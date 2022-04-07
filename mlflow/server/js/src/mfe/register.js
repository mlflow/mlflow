// TODO Once module sharing is ready, export this and remove 'commitHash' below
// export { commitHash } from '../../infra/CommitHash';

export function commitHash() {
  return process.env.GIT_COMMIT_HASH;
}
