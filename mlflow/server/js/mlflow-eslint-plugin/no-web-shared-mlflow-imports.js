const path = require('path');

const MLFLOW_PACKAGE_NAME = '@mlflow/mlflow';
const WEB_SHARED_PATH_SEGMENT = `${path.sep}src${path.sep}shared${path.sep}web-shared${path.sep}`;

const isPathWithinDirectory = (childPath, parentDirectory) => {
  const relativePath = path.relative(parentDirectory, childPath);
  return relativePath === '' || (!relativePath.startsWith('..') && !path.isAbsolute(relativePath));
};

const getWebSharedContext = (normalizedFilename) => {
  const segmentIndex = normalizedFilename.indexOf(WEB_SHARED_PATH_SEGMENT);
  if (segmentIndex === -1) {
    return null;
  }

  const jsRoot = normalizedFilename.slice(0, segmentIndex);
  return {
    jsRoot,
    webSharedRoot: path.join(jsRoot, 'src', 'shared', 'web-shared'),
  };
};

const isMlflowImport = (importPath) =>
  importPath === MLFLOW_PACKAGE_NAME || importPath.startsWith(`${MLFLOW_PACKAGE_NAME}/`);

module.exports = {
  meta: {
    type: 'problem',
    docs: {
      description: 'Disallow imports from @mlflow/mlflow in web-shared source',
      category: 'Best Practices',
      recommended: true,
    },
    schema: [],
    messages: {
      noMlflowImportsInWebShared:
        "Do not import from '@mlflow/mlflow' (or equivalent relative paths) inside src/shared/web-shared.",
    },
  },
  create(context) {
    const filename = context.getFilename();
    const normalizedFilename = path.normalize(filename);
    const webSharedContext = getWebSharedContext(normalizedFilename);
    if (!webSharedContext) {
      return {};
    }

    const reportIfRestricted = (node, importPath) => {
      if (typeof importPath !== 'string') {
        return;
      }

      if (isMlflowImport(importPath)) {
        context.report({
          node,
          messageId: 'noMlflowImportsInWebShared',
        });
        return;
      }

      if (!importPath.startsWith('.')) {
        return;
      }

      const resolvedImportPath = path.resolve(path.dirname(normalizedFilename), importPath);
      if (
        isPathWithinDirectory(resolvedImportPath, webSharedContext.jsRoot) &&
        !isPathWithinDirectory(resolvedImportPath, webSharedContext.webSharedRoot)
      ) {
        context.report({
          node,
          messageId: 'noMlflowImportsInWebShared',
        });
      }
    };

    return {
      ImportDeclaration(node) {
        reportIfRestricted(node.source, node.source.value);
      },
      ExportAllDeclaration(node) {
        if (node.source) {
          reportIfRestricted(node.source, node.source.value);
        }
      },
      ExportNamedDeclaration(node) {
        if (node.source) {
          reportIfRestricted(node.source, node.source.value);
        }
      },
      ImportExpression(node) {
        if (node.source?.type === 'Literal') {
          reportIfRestricted(node.source, node.source.value);
        }
      },
    };
  },
};
