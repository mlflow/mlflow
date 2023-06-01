/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { ArtifactNode } from './ArtifactUtils';

const getTestArtifactNode = () => {
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
  const rootNode = new ArtifactNode(true, undefined);
  rootNode.isLoaded = true;
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
  const dir1 = new ArtifactNode(false, { path: 'dir1', is_dir: true });
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
  const file1 = new ArtifactNode(false, { path: 'file1', is_dir: false, file_size: '159' });
  rootNode.children = { dir1, file1 };
  return rootNode;
};

test('deepCopy works properly', () => {
  const rootNode = getTestArtifactNode();
  const copiedNode = rootNode.deepCopy();
  // Checks equality of all members.
  expect(rootNode).toEqual(copiedNode);
  expect(rootNode).not.toBe(copiedNode);
});
