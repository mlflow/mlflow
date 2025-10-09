/**
 * ESLint rule to validate API Reference URLs.
 *
 * This rule ensures that links to /api_reference/ use the <APILink> component.
 *
 * @type {import('eslint').Rule.RuleModule}
 */
module.exports = {
  meta: {
    type: 'suggestion',
    docs: {
      description: 'Detect `Link` components with `to` props to API references and suggest using APILink',
      category: 'Best Practices',
    },
    schema: [],
    messages: {
      useAPILink: 'Use the <APILink> component for API reference links',
    },
  },

  /**
   * Creates the rule implementation.
   *
   * @param {import('eslint').Rule.RuleContext} context - The ESLint rule context
   * @returns {import('eslint').Rule.RuleListener} The rule visitor methods
   */
  create(context) {
    return {
      /**
       * Validates JSX opening elements for `Link` tags.
       *
       * @param {import('estree').Node} node - The JSX opening element node
       */
      JSXOpeningElement(node) {
        if (node.type !== 'JSXOpeningElement' || node.name.type !== 'JSXIdentifier' || node.name.name !== 'Link') {
          return;
        }
        const toAttr = node.attributes.find((attr) => attr.type === 'JSXAttribute' && attr.name.name === 'to');

        if (!toAttr) {
          return;
        }

        const toValue = getHrefValue(toAttr);

        if (!toValue) {
          // Can't determine a static value.
          return;
        }

        if (toValue.startsWith('/api_reference/')) {
          context.report({
            node,
            messageId: 'useAPILink',
          });
        }
      },
    };
  },
};

/**
 * Extracts the href value from a JSX attribute.
 *
 * @param {import('estree-jsx').JSXAttribute} attr - The JSX attribute node
 * @returns {string|null} The href value or null if it can't be determined
 */
function getHrefValue(attr) {
  if (!attr.value) return null;

  if (attr.value.type === 'Literal') {
    return attr.value.value;
  }

  if (attr.value.type === 'JSXExpressionContainer' && attr.value.expression.type === 'Literal') {
    return attr.value.expression.value;
  }

  if (
    attr.value.type === 'JSXExpressionContainer' &&
    attr.value.expression.type === 'TemplateLiteral' &&
    attr.value.expression.expressions.length === 0
  ) {
    return attr.value.expression.quasis[0].value.raw;
  }

  // Can't determine value for dynamic expressions
  return null;
}
