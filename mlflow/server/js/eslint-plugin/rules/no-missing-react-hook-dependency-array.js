const hooksWithDependencyArrays = new Set(['useCallback', 'useEffect', 'useMemo']);

module.exports = {
  meta: {
    type: 'problem',
    messages: {
      missingDependencyArray:
        'Did you mean to call {{ functionName }} without a dependency array? This is an advanced way to use this hook, please check code carefully.',
    },
  },
  create(context) {
    return {
      CallExpression(node) {
        let functionName;

        switch (node.callee.type) {
          case 'Identifier': {
            // This is the "useEffect" case
            functionName = node.callee.name;
            break;
          }
          case 'MemberExpression': {
            // This is the "React.useEffect" case
            functionName = node.callee.property.name;
            break;
          }
          default: {
            break;
          }
        }

        // Check if the function is a hook and only has a function and no dep array
        if (functionName !== undefined && hooksWithDependencyArrays.has(functionName) && node.arguments.length === 1) {
          context.report({
            data: { functionName },
            node,
            messageId: 'missingDependencyArray',
          });
        }
      },
    };
  },
};
