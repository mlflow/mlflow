module.exports = {
  meta: {
    type: 'problem',
    messages: {
      noHardcodedColors: 'Use colors from the @databricks/design-system theme instead of hardcoded colors',
    },
  },
  create(context) {
    return {
      Literal(node) {
        const patterns = [/^#[a-f0-9]{3,4}$/i, /^#[a-f0-9]{6}$/i, /^#[a-f0-9]{8}$/i, /^rgba?\(/i, /^white$/];
        if (patterns.some((pattern) => pattern.test(node.value))) {
          context.report({
            node,
            messageId: 'noHardcodedColors',
          });
        }
      },
    };
  },
};
