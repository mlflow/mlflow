/**
 * ESLint rule to detect raw image paths that should use useBaseUrl in Docusaurus.
 *
 * This rule ensures that image paths in JSX elements use the useBaseUrl hook
 * instead of raw paths to prevent broken links in the published site.
 *
 * @type {import('eslint').Rule.RuleModule}
 */
module.exports = {
  meta: {
    type: 'problem',
    docs: {
      description: 'Detect raw image paths that should use useBaseUrl in Docusaurus',
      category: 'Possible Errors',
    },
    fixable: null,
    schema: [],
    messages: {
      rawImagePath:
        "Raw image path \"{{path}}\" will break in production. Use one of these methods:\n\n1. Import as ES6 module:\n  import ImageUrl from '@site/static{{path}}';\n  <img src={ImageUrl} />\n\n2. Use useBaseUrl:\n  import useBaseUrl from '@docusaurus/useBaseUrl';\n  <img src={useBaseUrl('{{path}}')} />\n\nSee: https://docusaurus.io/docs/static-assets#referencing-your-static-asset",
    },
  },

  /**
   * Creates the rule implementation.
   *
   * @param {import('eslint').Rule.RuleContext} context - The ESLint rule context
   * @returns {import('eslint').Rule.RuleListener} The rule visitor methods
   */
  create(context) {
    /**
     * Extracts the string value from various node types.
     *
     * @param {import('estree').Node} node - The node to extract value from
     * @returns {string|null} The extracted string value or null
     */
    function getStringValue(node) {
      if (!node) return null;

      // Handle literal strings
      if (node.type === 'Literal' && typeof node.value === 'string') {
        return node.value;
      }

      // Handle template literals without expressions
      if (node.type === 'TemplateLiteral' && node.expressions.length === 0) {
        return node.quasis[0].value.raw;
      }

      return null;
    }

    return {
      /**
       * Validates JSX opening elements for img tags and Image components.
       *
       * @param {import('estree').Node} node - The JSX opening element node
       */
      JSXOpeningElement(node) {
        if (node.type !== 'JSXOpeningElement') return;

        const elementName = node.name.type === 'JSXIdentifier' ? node.name.name : null;

        // Check if this is an img tag or Image component
        if (elementName !== 'img' && elementName !== 'Image') return;

        // Find the src attribute
        const srcAttr = node.attributes.find((attr) => attr.type === 'JSXAttribute' && attr.name.name === 'src');

        if (!srcAttr || !srcAttr.value) return;

        let srcValue = null;

        // Handle different attribute value types
        if (srcAttr.value.type === 'Literal') {
          srcValue = srcAttr.value.value;
        } else if (srcAttr.value.type === 'JSXExpressionContainer') {
          srcValue = getStringValue(srcAttr.value.expression);
        }

        // Check if it's a raw static path
        if (srcValue && srcValue.startsWith('/')) {
          context.report({
            node: srcAttr,
            messageId: 'rawImagePath',
            data: { path: srcValue },
          });
        }
      },
    };
  },
};
