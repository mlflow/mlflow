/**
 * Custom ESLint rule to enforce that all static componentId values are
 * registered in the componentId registry.
 *
 * This ensures a curated, auditable inventory of every componentId used
 * in the MLflow UI. Dynamic componentIds (template literals with expressions,
 * variable references) are skipped since they cannot be statically resolved.
 */

const registry = require('./componentId-registry');

module.exports = {
  meta: {
    type: 'problem',
    docs: {
      description: 'Require componentId values to be registered in the componentId registry',
      category: 'Best Practices',
      recommended: true,
    },
    fixable: null,
    schema: [],
    messages: {
      unregisteredComponentId:
        'componentId "{{id}}" is not in the registry. Add it to mlflow-eslint-plugin/componentId-registry.js.',
    },
  },
  create(context) {
    /**
     * Walk up from a node to determine if it is used as a componentId value.
     * Handles:
     *   - <Foo componentId="value" />
     *   - <foo data-component-id="value" />
     *   - <Foo componentId={"value"} />
     *   - <Foo componentId={cond ? "a" : "b"} />
     *   - { componentId: "value" }
     */
    function isComponentIdContext(node) {
      let current = node.parent;

      // Walk through wrappers: JSXExpressionContainer, ConditionalExpression
      while (current) {
        if (current.type === 'JSXExpressionContainer') {
          current = current.parent;
          continue;
        }
        if (current.type === 'ConditionalExpression') {
          current = current.parent;
          continue;
        }
        break;
      }

      if (!current) return false;

      // JSX attribute: componentId="value" or data-component-id="value"
      if (current.type === 'JSXAttribute') {
        const name =
          current.name.type === 'JSXIdentifier'
            ? current.name.name
            : current.name.type === 'JSXNamespacedName'
              ? `${current.name.namespace.name}:${current.name.name.name}`
              : '';
        return name === 'componentId' || name === 'data-component-id';
      }

      // Object property: { componentId: "value" }
      if (
        current.type === 'Property' &&
        current.key.type === 'Identifier' &&
        current.key.name === 'componentId' &&
        current.value === node // ensure the string is the value, not the key
      ) {
        return true;
      }

      return false;
    }

    function checkValue(node, value) {
      if (typeof value !== 'string') return;
      if (!isComponentIdContext(node)) return;

      if (!Object.prototype.hasOwnProperty.call(registry, value)) {
        context.report({
          node,
          messageId: 'unregisteredComponentId',
          data: { id: value },
        });
      }
    }

    return {
      Literal(node) {
        checkValue(node, node.value);
      },
      TemplateLiteral(node) {
        // Only check fully static template literals (no expressions)
        if (node.expressions.length === 0 && node.quasis.length === 1) {
          checkValue(node, node.quasis[0].value.cooked);
        }
      },
    };
  },
};
