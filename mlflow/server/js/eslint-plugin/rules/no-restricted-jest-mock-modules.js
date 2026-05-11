const ERROR = 'NO_RESTRICTED_JEST_MOCK_MODULES';

const RESTRICTED_MODULES = {
  '@emotion/react': '',
  '@databricks/design-system': 'Did you forget to wrap your component in a `DesignSystemProvider`?',
};

module.exports = {
  meta: {
    docs: {
      description: 'Disallow mocking certain modules.',
    },
    messages: {
      [ERROR]: '{{ moduleName }} should not be mocked. {{ additionalInfo }}',
    },
  },

  create(context) {
    return {
      CallExpression(node) {
        if (
          node.callee.type === 'MemberExpression' &&
          node.callee.object.name === 'jest' &&
          node.callee.property.name === 'mock' &&
          Object.keys(RESTRICTED_MODULES).includes(node.arguments[0].value)
        ) {
          const moduleName = node.arguments[0].value;
          context.report({
            data: {
              moduleName,
              additionalInfo: RESTRICTED_MODULES[moduleName],
            },
            node,
            messageId: ERROR,
          });
        }
      },
    };
  },
};
