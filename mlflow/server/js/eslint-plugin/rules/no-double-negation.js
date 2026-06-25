module.exports = {
  meta: {
    type: 'problem',
    messages: {
      noDoubleNegation: 'Use Boolean(<value>) instead of !!<value> when casting to a boolean value.',
    },
    fixable: 'code',
  },
  create(context) {
    const sourceCode = context.getSourceCode();

    return {
      UnaryExpression(node) {
        if (node.operator === '!' && node.parent.type === 'UnaryExpression' && node.parent.operator === '!') {
          context.report({
            node,
            messageId: 'noDoubleNegation',
            fix(fixer) {
              return fixer.replaceText(node.parent, `Boolean(${sourceCode.getText(node.argument)})`);
            },
          });
        }
      },
    };
  },
};
