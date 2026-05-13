const ERROR = 'NO_RESTRICTED_GLOBALS_WITH_MODULE';

module.exports = {
  meta: {
    fixable: 'code',
    messages: {
      [ERROR]: `Do not use global {{ name }}. Import it from {{ module }}: import { {{name}} } from '{{ module }}';`,
    },
    schema: [
      // Pass in an object that maps the restricted global identifier to the module
      // from which the identifier should be imported. Example: `{ expect: 'chai', it: '@jest/globals' }`.
      {
        type: 'object',
        additionalProperties: { type: 'string' },
      },
    ],
  },
  create(context) {
    const restrictedGlobalsNameToModule = context.options[0];

    /* Checks whether an identifier is defined somewhere in the module but not the global scope. */
    function isDefinedInScope(name, scope) {
      if (scope.type === 'global') return false;
      const isDefined = scope.variables.some((variable) => variable.name === name);
      if (isDefined) return true;
      if (scope.upper) return isDefinedInScope(name, scope.upper);
      return false;
    }

    let lastImportDeclarationNode = null;

    return {
      ImportDeclaration(node) {
        lastImportDeclarationNode = node;
      },

      'CallExpression, MemberExpression'(node) {
        const name = node.type === 'CallExpression' ? node.callee.name : node.object.name;

        if (!name) return;

        if (!restrictedGlobalsNameToModule.hasOwnProperty(name)) return;

        const isDefined = isDefinedInScope(name, context.getScope());

        if (isDefined) return;

        const module = restrictedGlobalsNameToModule[name];

        context.report({
          loc: {
            start: {
              column: node.loc.start.column,
              line: node.loc.start.line,
            },
            end: {
              column: node.loc.start.column + name.length,
              line: node.loc.start.line,
            },
          },
          data: {
            name,
            module,
          },
          messageId: ERROR,
          fix(fixer) {
            const newImport = `import { ${name} } from '${module}';\n`;
            if (lastImportDeclarationNode) {
              // insert after the last import decl
              return fixer.insertTextAfter(lastImportDeclarationNode, newImport);
            }
            // insert at the start of the file
            return fixer.insertTextAfterRange([0, 0], newImport);
          },
        });
      },
    };
  },
};
