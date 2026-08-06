const BANNED_PROPERTIES = ['location', 'history'];

module.exports = {
  meta: {
    type: 'problem',
    messages: {
      noMock:
        'Object.defineProperty(window, "{{ property }}", ...) is forbidden. Use a test router utility instead of directly mocking window.location or window.history.',
    },
  },
  create(context) {
    return {
      "CallExpression[callee.object.name='Object'][callee.property.name='defineProperty']"(node) {
        if (node.arguments[0].name === 'window' && BANNED_PROPERTIES.includes(node.arguments[1].value)) {
          context.report({
            node,
            data: {
              property: node.arguments[1].value,
            },
            messageId: 'noMock',
          });
        }
      },
    };
  },
};
