import { ArtifactNode } from './ArtifactUtils';

const getTestArtifactNode = () => {
  const rootNode = new ArtifactNode(true, undefined);
  rootNode.isLoaded = true;
  const dir1 = new ArtifactNode(false, { path: 'dir1', is_dir: true });
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
