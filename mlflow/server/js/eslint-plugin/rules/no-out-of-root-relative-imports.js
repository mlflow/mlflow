const path = require('path');

// Allow JSON files since they usually import data and not source files
const allowedFileExtensions = ['.json'];

module.exports = {
  meta: {
    type: 'problem',
    messages: {
      outOfRootRelativeImport:
        'Importing outside of package root is not allowed. Please import from within the package root.',
      relativeNodeModuleImport: 'Using relative paths to import from within a package is not allowed.',
    },
  },
  create(context) {
    return {
      ImportDeclaration(node) {
        if (node.source.value.startsWith('.')) {
          // Consider imports starting with a '.' as relative and just resolve the path directly
          const resolvedPath = path.resolve(context.filename, node.source.value);

          if (allowedFileExtensions.includes(path.extname(resolvedPath))) {
            return;
          }

          const isImportWithinCwd = isWithin(context.cwd, resolvedPath);

          if (!isImportWithinCwd) {
            context.report({
              node,
              messageId: 'outOfRootRelativeImport',
            });
          }
        } else if (node.source.value.includes('..')) {
          // Non-relative imports that contain a '..' are importing relatively inside of a module
          context.report({
            node,
            messageId: 'relativeNodeModuleImport',
          });
        }
      },
    };
  },
};

function isWithin(outer, inner) {
  const rel = path.relative(outer, inner);
  return !rel.startsWith('../') && rel !== '..';
}
