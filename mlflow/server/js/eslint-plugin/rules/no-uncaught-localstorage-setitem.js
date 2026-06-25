module.exports = {
  create(context) {
    return {
      MemberExpression(node) {
        const isSetItemCall =
          node.property.name === 'setItem' &&
          ((node.object.type === 'MemberExpression' &&
            node.object.object.name === 'window' &&
            node.object.property.name === 'localStorage') ||
            (node.object.type === 'Identifier' && node.object.name === 'localStorage'));
        if (!isSetItemCall) {
          return;
        }

        let current = node;
        let foundTryCatch = false;
        while (current.parent) {
          current = current.parent;
          if (current.type === 'TryStatement') {
            foundTryCatch = true;
            break;
          }
        }

        if (!foundTryCatch) {
          context.report({
            node,
            message:
              'localStorage.setItem may throw QuotaExceededError. Wrap the call in a try-catch block to handle the error.',
          });
        }
      },
    };
  },
};
