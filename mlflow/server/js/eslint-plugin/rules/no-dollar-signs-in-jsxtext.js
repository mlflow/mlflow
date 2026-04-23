module.exports = {
  create(context) {
    return {
      JSXText(node) {
        if (node.raw.indexOf('$') !== -1) {
          context.report(node, '"$" is disallowed; use "&#36;" to display the $-symbol.');
        }
      },
      Literal(node) {
        // workaround for https://github.com/babel/babel-eslint/pull/785
        if (node.raw.endsWith('$')) context.report(node, '"$" is disallowed as the end of a Literal.');
      },
    };
  },
};
