import type { API, FileInfo } from 'jscodeshift';

module.exports = function (file: FileInfo, api: API) {
  const j = api.jscodeshift;
  const root = j(file.source);

  // Replace all imports of Skeleton with LegacySkeleton
  const skeletonsFound = root
    .find(j.ImportDeclaration)
    .filter((path) => path.node.source.value === '@databricks/design-system')
    .find(j.ImportSpecifier)
    .filter((path) => path.node.imported.name === 'Skeleton');

  skeletonsFound.replaceWith(j.importSpecifier(j.identifier('LegacySkeleton')));

  // Find instances of Skeleton
  const skeletonElements = root.findJSXElements('Skeleton');
  // Rename each "Skeleton" to "LegacySkeleton"
  skeletonElements.forEach((element) => {
    j(element).replaceWith(
      j.jsxElement(
        // true makes the element self-closing
        j.jsxOpeningElement(j.jsxIdentifier('LegacySkeleton'), element.node.openingElement.attributes, true),
        null,
        [],
      ),
    );
  });

  return root.toSource();
};
