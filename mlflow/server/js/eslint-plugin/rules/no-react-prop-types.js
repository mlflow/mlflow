module.exports = {
  create: (context) => {
    const sourceCode = context.getSourceCode();

    function report(node) {
      context.report({
        node: node,
        message: 'React.PropTypes is deprecated; use the npm module prop-types instead',
      });
    }

    return {
      MemberExpression: (node) => {
        if (sourceCode.getText(node) === 'React.PropTypes') report(node);
      },
      ImportDeclaration: (node) => {
        if (node.source.value !== 'react') return;
        node.specifiers.forEach((specifier) => {
          if (specifier.imported && specifier.imported.name === 'PropTypes') report(node);
        });
      },
    };
  },
};
