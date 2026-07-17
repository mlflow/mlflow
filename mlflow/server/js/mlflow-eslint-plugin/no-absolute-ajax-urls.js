/**
 * Custom ESLint rule to prevent absolute AJAX URLs in the UI, as some deployments
 * configurations will break if the AJAX URL is absolute.
 *
 * All AJAX routes should be relative and wrapped in `getAjaxUrl`, which automatically
 * appends a leading slash based on whether a build-time env var (`MLFLOW_USE_ABSOLUTE_AJAX_URLS`)
 * is set.
 */

module.exports = {
  meta: {
    type: 'problem',
    docs: {
      description: 'Disallow absolute AJAX URLs containing /ajax-api/',
      category: 'Best Practices',
      recommended: true,
    },
    fixable: null,
    schema: [],
    messages: {
      absoluteAjaxUrl:
        'Absolute AJAX URL detected. Use relative URLs wrapped in getAjaxUrl() instead. Example: getAjaxUrl("ajax-api/...")',
    },
  },
  create(context) {
    return {
      Literal(node) {
        // Check if the node is a string literal
        if (typeof node.value !== 'string') {
          return;
        }

        const stringValue = node.value;

        // Check if the string contains '/ajax-api/'
        if (stringValue.includes('/ajax-api/')) {
          context.report({
            node,
            messageId: 'absoluteAjaxUrl',
          });
        }
      },
      TemplateLiteral(node) {
        // Check template literals (e.g., `something ${var}`)
        // We check the quasis (static parts of the template)
        for (const quasi of node.quasis) {
          if (quasi.value.raw.includes('/ajax-api/')) {
            context.report({
              node,
              messageId: 'absoluteAjaxUrl',
            });
          }
        }
      },
    };
  },
};
