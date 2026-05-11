const ERROR = 'NO_UNTYPED_JEST_REQUIRE_ACTUAL';

module.exports = {
  meta: {
    docs: {
      description:
        'Disallow using jest.requireActual() without an explicit type parameter. This allows the return value to be spread like an object without TypeScript complaining.',
    },
    fixable: 'code',
    messages: {
      [ERROR]: 'jest.requireActual must be passed a type parameter.',
    },
  },

  create(context) {
    // This rule only applies to TypeScript files
    if (context.filename.endsWith('.js') || context.filename.endsWith('.jsx')) {
      return {};
    }

    return {
      CallExpression(node) {
        if (
          node.callee.type === 'MemberExpression' &&
          node.callee.object.name === 'jest' &&
          node.callee.property.name === 'requireActual' &&
          node.typeArguments === undefined
        ) {
          const requireActualImport = node.arguments[0].value;
          context.report({
            fix(fixer) {
              return fixer.insertTextAfter(node.callee, `<typeof import('${requireActualImport}')>`);
            },
            node,
            messageId: ERROR,
          });
        }
      },
    };
  },
};
