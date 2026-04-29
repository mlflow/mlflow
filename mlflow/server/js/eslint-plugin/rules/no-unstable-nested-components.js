/**
 * Forked from https://github.com/jsx-eslint/eslint-plugin-react/blob/d50d8865210d16b46a9bb0a531b19cef42eab133/lib/rules/no-unstable-nested-components.js
 * Licensed under MIT
 */

const Components = require('eslint-plugin-react/lib/util/Components');
const isCreateElement = require('eslint-plugin-react/lib/util/isCreateElement');
const report = require('eslint-plugin-react/lib/util/report');

// ------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------

const HOOK_REGEXP = /^use[A-Z0-9].*$/;

// ------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------

/**
 * Generate error message with given parent component name
 * @param {String} parentName Name of the parent component, if known
 * @returns {String} Error message with parent component name
 */
function generateErrorMessageWithParentName(parentName) {
  return `Do not define components during render. React will see a new component type on every render and destroy the entire subtree's DOM nodes and state (https://reactjs.org/docs/reconciliation.html#elements-of-different-types). Instead, move this component definition out of the parent component${
    parentName ? ` “${parentName}” ` : ' '
  }and pass data as props.`;
}

/**
 * Check whether given text starts with `render`. Comparison is case-sensitive.
 * @param {String} text Text to validate
 * @returns {Boolean}
 */
function startsWithRender(text) {
  return (text || '').startsWith('render');
}

/**
 * Get closest parent matching given matcher
 * @param {ASTNode} node The AST node
 * @param {Context} context eslint context
 * @param {Function} matcher Method used to match the parent
 * @returns {ASTNode} The matching parent node, if any
 */
function getClosestMatchingParent(node, context, matcher) {
  if (!node || !node.parent || node.parent.type === 'Program') {
    return;
  }

  if (matcher(node.parent, context)) {
    return node.parent;
  }

  return getClosestMatchingParent(node.parent, context, matcher);
}

/**
 * Matcher used to check whether given node is a `createElement` call
 * @param {ASTNode} node The AST node
 * @param {Context} context eslint context
 * @returns {Boolean} True if node is a `createElement` call, false if not
 */
function isCreateElementMatcher(node, context) {
  return node && node.type === 'CallExpression' && isCreateElement(node, context);
}

/**
 * Matcher used to check whether given node is a `ObjectExpression`
 * @param {ASTNode} node The AST node
 * @returns {Boolean} True if node is a `ObjectExpression`, false if not
 */
function isObjectExpressionMatcher(node) {
  return node && node.type === 'ObjectExpression';
}

/**
 * Matcher used to check whether given node is a `JSXExpressionContainer`
 * @param {ASTNode} node The AST node
 * @returns {Boolean} True if node is a `JSXExpressionContainer`, false if not
 */
function isJSXExpressionContainerMatcher(node) {
  return node && node.type === 'JSXExpressionContainer';
}

/**
 * Matcher used to check whether given node is a `JSXElement`
 * @param {ASTNode} node The AST node
 * @returns {Boolean} True if node is a `JSXExpressionContainer`, false if not
 */
function isJSXElementMatcher(node) {
  return node && node.type === 'JSXElement';
}

/**
 * Matcher used to check whether given node is a `JSXAttribute` of `JSXExpressionContainer`
 * @param {ASTNode} node The AST node
 * @returns {Boolean} True if node is a `JSXAttribute` of `JSXExpressionContainer`, false if not
 */
function isJSXAttributeOfExpressionContainerMatcher(node) {
  return node && node.type === 'JSXAttribute' && node.value && node.value.type === 'JSXExpressionContainer';
}

/**
 * Matcher used to check whether given node is an object `Property`
 * @param {ASTNode} node The AST node
 * @returns {Boolean} True if node is a `Property`, false if not
 */
function isPropertyOfObjectExpressionMatcher(node) {
  return node && node.parent && node.parent.type === 'Property';
}

/**
 * Matcher used to check whether given node is a `CallExpression`
 * @param {ASTNode} node The AST node
 * @returns {Boolean} True if node is a `CallExpression`, false if not
 */
function isCallExpressionMatcher(node) {
  return node && node.type === 'CallExpression';
}

/**
 * Check whether given node or its parent is directly inside `map` call
 * ```jsx
 * {items.map(item => <li />)}
 * ```
 * @param {ASTNode} node The AST node
 * @returns {Boolean} True if node is directly inside `map` call, false if not
 */
function isMapCall(node) {
  return node && node.callee && node.callee.property && node.callee.property.name === 'map';
}

/**
 * Check whether given node is `ReturnStatement` of a React hook
 * @param {ASTNode} node The AST node
 * @param {Context} context eslint context
 * @returns {Boolean} True if node is a `ReturnStatement` of a React hook, false if not
 */
function isReturnStatementOfHook(node, context) {
  if (!node || !node.parent || node.parent.type !== 'ReturnStatement') {
    return false;
  }

  const callExpression = getClosestMatchingParent(node, context, isCallExpressionMatcher);
  return callExpression && callExpression.callee && HOOK_REGEXP.test(callExpression.callee.name);
}

/**
 * Check whether given node is declared inside a render prop
 * ```jsx
 * <Component renderFooter={() => <div />} />
 * <Component>{() => <div />}</Component>
 * ```
 * @param {ASTNode} node The AST node
 * @param {Context} context eslint context
 * @returns {Boolean} True if component is declared inside a render prop, false if not
 */
function isComponentInRenderProp(node, context) {
  if (
    node &&
    node.parent &&
    node.parent.type === 'Property' &&
    node.parent.key &&
    startsWithRender(node.parent.key.name)
  ) {
    return true;
  }

  // Check whether component is a render prop used as direct children, e.g. <Component>{() => <div />}</Component>
  if (
    node &&
    node.parent &&
    node.parent.type === 'JSXExpressionContainer' &&
    node.parent.parent &&
    node.parent.parent.type === 'JSXElement'
  ) {
    return true;
  }

  const jsxExpressionContainer = getClosestMatchingParent(node, context, isJSXExpressionContainerMatcher);

  // Check whether prop name indicates accepted patterns
  if (
    jsxExpressionContainer &&
    jsxExpressionContainer.parent &&
    jsxExpressionContainer.parent.type === 'JSXAttribute' &&
    jsxExpressionContainer.parent.name &&
    jsxExpressionContainer.parent.name.type === 'JSXIdentifier'
  ) {
    const propName = jsxExpressionContainer.parent.name.name;

    // Starts with render, e.g. <Component renderFooter={() => <div />} />
    if (startsWithRender(propName)) {
      return true;
    }

    // Uses children prop explicitly, e.g. <Component children={() => <div />} />
    if (propName === 'children') {
      return true;
    }
  }

  return false;
}

/**
 * Check whether given node is declared directly inside a render property
 * ```jsx
 * const rows = { render: () => <div /> }
 * <Component rows={ [{ render: () => <div /> }] } />
 *  ```
 * @param {ASTNode} node The AST node
 * @returns {Boolean} True if component is declared inside a render property, false if not
 */
function isDirectValueOfRenderProperty(node) {
  return (
    node &&
    node.parent &&
    node.parent.type === 'Property' &&
    node.parent.key &&
    node.parent.key.type === 'Identifier' &&
    startsWithRender(node.parent.key.name)
  );
}

/**
 * Resolve the component name of given node
 * @param {ASTNode} node The AST node of the component
 * @returns {String} Name of the component, if any
 */
function resolveComponentName(node) {
  const parentName = node.id && node.id.name;
  if (parentName) return parentName;

  return node.type === 'ArrowFunctionExpression' && node.parent && node.parent.id && node.parent.id.name;
}

/**
 * Returns the name of the function, which is being called in the given node.
 *
 * If the function is a member of an object, the property name will be returned and the
 * actual object name will be ignored.
 *
 * @param {ASTNode} node
 * @returns {String} Name of the function being called in the node
 */
function getFunctionCallee(node) {
  const callee = node.callee;
  if (callee.type === 'MemberExpression') {
    return callee.property.name;
  } else if (callee.type === 'Identifier') {
    return callee.name;
  }
}

/**
 * Check whether given node is a styled-components call. Detects the following patterns:
 *
 * - styled.div`...`
 * - styled.div(...)
 * - styled(Component)``
 * - styled(Component)(...)
 *
 * @param {ASTNode} node The AST node
 * @returns {Boolean} True if node is a styled-components call, false if not
 */
function isStyledComponentsCall(node) {
  function expressionRefersToStyled(expression) {
    if (
      // styled.div`...` or styled.div(...)
      expression.type === 'MemberExpression' &&
      expression.object?.type === 'Identifier' &&
      expression.object?.name === 'styled'
    ) {
      return true;
    }

    if (
      // styled(Div)`...` or styled(Div)(...)
      expression.type === 'CallExpression' &&
      expression.callee?.type === 'Identifier' &&
      expression.callee?.name === 'styled'
    ) {
      return true;
    }

    return false;
  }

  if (!node) {
    return false;
  } else if (node.type === 'TaggedTemplateExpression') {
    return expressionRefersToStyled(node.tag);
  } else if (node.type === 'CallExpression') {
    return expressionRefersToStyled(node.callee);
  } else {
    return false;
  }
}

// ------------------------------------------------------------------------------
// Rule Definition
// ------------------------------------------------------------------------------

/** @type {import('eslint').Rule.RuleModule} */
module.exports = {
  meta: {
    docs: {
      description: 'Disallow creating unstable components inside components',
      category: 'Possible Errors',
      recommended: false,
      url: 'https://reactjs.org/docs/reconciliation.html#elements-of-different-types',
    },
    schema: [
      {
        type: 'object',
        properties: {
          allowAsProps: {
            type: 'boolean',
          },
          allowAsPropsInElements: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                name: { type: 'string' },
                props: { type: 'array', items: { type: 'string' } },
              },
            },
          },
          allowAsPropsInFunctionCalls: {
            type: 'array',
            items: {
              type: 'string',
            },
          },
        },
        additionalProperties: false,
      },
    ],
  },

  create: Components.detect((context, components, utils) => {
    const allowAsProps = context.options.some((option) => option && option.allowAsProps);
    // collect the allowed elements names and associated attribuites if any
    const allowedAsPropsInElements = context.options
      .filter((option) => option && option.allowAsPropsInElements)
      .map((o) => o.allowAsPropsInElements)
      .flat();
    const allowedAsPropsInFunctionCalls = context.options
      .filter((option) => option && option.allowAsPropsInFunctionCalls)
      .map((o) => o.allowAsPropsInFunctionCalls)
      .flat();

    /**
     * Check whether given node is declared inside class component's render block
     * ```jsx
     * class Component extends React.Component {
     *   render() {
     *     class NestedClassComponent extends React.Component {
     * ...
     * ```
     * @param {ASTNode} node The AST node being checked
     * @returns {Boolean} True if node is inside class component's render block, false if not
     */
    function isInsideRenderMethod(node) {
      const parentComponent = utils.getParentComponent(node);

      if (!parentComponent || parentComponent.type !== 'ClassDeclaration') {
        return false;
      }

      return (
        node &&
        node.parent &&
        node.parent.type === 'MethodDefinition' &&
        node.parent.key &&
        node.parent.key.name === 'render'
      );
    }

    /**
     * Check whether given node is a function component declared inside class component.
     * Util's component detection fails to detect function components inside class components.
     * ```jsx
     * class Component extends React.Component {
     *  render() {
     *    const NestedComponent = () => <div />;
     * ...
     * ```
     * @param {ASTNode} node The AST node being checked
     * @returns {Boolean} True if given node a function component declared inside class component, false if not
     */
    function isFunctionComponentInsideClassComponent(node) {
      const parentComponent = utils.getParentComponent(node);
      const parentStatelessComponent = utils.getParentStatelessComponent(node);

      return (
        parentComponent &&
        parentStatelessComponent &&
        parentComponent.type === 'ClassDeclaration' &&
        utils.getStatelessComponent(parentStatelessComponent) &&
        utils.isReturningJSX(node)
      );
    }

    /**
     * Check whether given node is declared inside `createElement` call's props
     * ```js
     * React.createElement(Component, {
     *   footer: () => React.createElement("div", null)
     * })
     * ```
     * @param {ASTNode} node The AST node
     * @returns {Boolean} True if node is declare inside `createElement` call's props, false if not
     */
    function isComponentInsideCreateElementsProp(node) {
      if (!components.get(node)) {
        return false;
      }

      const createElementParent = getClosestMatchingParent(node, context, isCreateElementMatcher);

      return (
        createElementParent &&
        createElementParent.arguments &&
        createElementParent.arguments[1] === getClosestMatchingParent(node, context, isObjectExpressionMatcher)
      );
    }

    /**
     * Check whether given node is declared inside a component/object prop.
     * ```jsx
     * <Component footer={() => <div />} />
     * { footer: () => <div /> }
     * ```
     * @param {ASTNode} node The AST node being checked
     * @returns {Boolean} True if node is a component declared inside prop, false if not
     */
    function isComponentInProp(node) {
      if (isPropertyOfObjectExpressionMatcher(node)) {
        return utils.isReturningJSX(node);
      }

      const jsxAttribute = getClosestMatchingParent(node, context, isJSXAttributeOfExpressionContainerMatcher);

      if (!jsxAttribute) {
        return isComponentInsideCreateElementsProp(node);
      }

      return utils.isReturningJSX(node);
    }

    /**
     * Check whether the given node is declared inside a prop within an allowed function call. Given the following
     * config:
     * ```js
     * [{ allowAsPropsInFunctionCalls: ['formatMessage' ]}]
     * ```
     * And the following code:
     * ```jsx
     *  <PipelinesHelpTooltip
     *     title={intl.formatMessage(updateTypeMessages.updateTypeTooltip, {
     *       link: (chunks) => (
     *         <Typography.Link
     *           href={DbGuideUrls.getDbGuideUrl(DbGuideLinks.DELTA_LIVE_TABLES_UPDATE_TYPE_URL)}
     *           openInNewTab
     *         >
     *           {chunks}
     *         </Typography.Link>
     *       ),
     *     })}
     *   />
     * ```
     * The use case will be allowed and not flagged.
     *
     * @param {ASTNode} node
     * @returns {Boolean} True if node is a component declared inside a prop within an allowed function call, false if not
     */
    function isPropertyInAllowedCallExpression(node) {
      const callNode = getClosestMatchingParent(node, context, isCallExpressionMatcher);

      if (!callNode) {
        return false;
      }
      const functionName = getFunctionCallee(callNode);
      return allowedAsPropsInFunctionCalls.includes(functionName);
    }

    /**
     * This function is given a node and returns the path to the node in the form of a string delimited by periods.
     *
     * For example, given the following example if the node was at randomRender the path would be: foo.randomRender
     * ```jsx
     *  <ExampleComponent foo={{ randomRender: () => <Bar /> }} />
     * ```
     *
     * @param {ASTNode} node
     * @returns {string | undefined} The path to the node in the form of a string delimited by periods, or undefined if the path could not be fully determined
     */
    function getJsxNestedPropertyPath(node) {
      const path = [];
      let current = node;
      let foundJsxAttribute = false;

      while (current) {
        if (current.type === 'Property' || current.type === 'ObjectProperty') {
          path.unshift(current.key.name);
        } else if (current.type === 'JSXAttribute') {
          path.unshift(current.name.name);
          foundJsxAttribute = true;
          break;
        }
        current = current.parent;
      }

      // Only return a path if we found a JSXAttribute and have at least one segment
      return foundJsxAttribute && path.length > 0 ? path.join('.') : undefined;
    }

    /**
     * Gets the full component name including namespace (e.g., "Foo.Bar")
     * @param {ASTNode} jsxElement
     * @returns {string|undefined}
     */
    function getFullComponentName(jsxElement) {
      if (jsxElement.type !== 'JSXElement' || !jsxElement.openingElement) {
        return undefined;
      }
      const nameNode = jsxElement.openingElement.name;

      if (nameNode.type === 'JSXIdentifier') {
        return nameNode.name;
      }

      if (nameNode.type === 'JSXMemberExpression') {
        const parts = [];
        let current = nameNode;

        while (current) {
          if (current.type === 'JSXIdentifier') {
            parts.unshift(current.name);
            break;
          }
          if (current.type === 'JSXMemberExpression') {
            parts.unshift(current.property.name);
            current = current.object;
          } else {
            break;
          }
        }

        return parts.join('.');
      }

      return undefined;
    }

    /**
     * Check whether the given node is declared inside a prop within an allowed JSX element. Given the following
     * config:
     * ```js
     * [{ allowAsPropsInElements: [{ name: 'FormattedMessage', attrs: ['values'] }]}]
     * ```
     * And the following code:
     * ```jsx
     *  <FormattedMessage
     *    defaultMessage="Email new users <excl>(excluding groups)</excl>"
     *    values={{
     *      excl: (chunks) => <span css={{ color: '#A3AEB8' }}>{chunks}</span>,
     *    }}
     *  />
     * ```
     *
     * ```js
     * [{ allowAsPropsInElements: ['foo.randomRender']}]
     * ```
     * And the following code:
     * ```jsx
     *  <ExampleComponent foo={{ randomRender: () => <Bar /> }} />
     * ```
     *
     * ```js
     * [{ allowAsPropsInElements: ['foo']}]
     * ```
     * And the following code:
     * ```jsx
     *  <ExampleComponent foo={{ randomRender: () => <Bar /> }} />
     * ```
     * These use cases will be allowed and not flagged.
     *
     * @param {ASTNode} node
     * @returns {Boolean} True if node is a component declared inside a prop within an allowed JSX element, false if not
     */
    function isPropertyInAllowedJsxElement(node) {
      const jsxElement = getClosestMatchingParent(node, context, isJSXElementMatcher);
      if (!jsxElement) {
        return false;
      }

      const elementName = getFullComponentName(jsxElement);
      const propertyPath = getJsxNestedPropertyPath(node);
      if (propertyPath === undefined) {
        return false;
      }

      return allowedAsPropsInElements.some(({ name, props }) => {
        if (name !== elementName) {
          return false;
        }

        return props.some((allowedProp) => propertyPath === allowedProp || propertyPath.startsWith(`${allowedProp}.`));
      });
    }

    /**
     * Check whether given node is a stateless component returning non-JSX
     * ```jsx
     * {{ a: () => null }}
     * ```
     * @param {ASTNode} node The AST node being checked
     * @returns {Boolean} True if node is a stateless component returning non-JSX, false if not
     */
    function isStatelessComponentReturningNull(node) {
      const component = utils.getStatelessComponent(node);

      return component && !utils.isReturningJSX(component);
    }

    /**
     * Check whether given node is a unstable nested component
     * @param {ASTNode} node The AST node being checked
     */
    function validate(node) {
      if (!node || !node.parent) {
        return;
      }

      const isDeclaredInsideProps = isComponentInProp(node);

      if (
        !components.get(node) &&
        !isFunctionComponentInsideClassComponent(node) &&
        !isDeclaredInsideProps &&
        !isStyledComponentsCall(node)
      ) {
        return;
      }

      if (
        // Support allowAsProps option
        (isDeclaredInsideProps &&
          (allowAsProps ||
            isComponentInRenderProp(node, context) ||
            isPropertyInAllowedJsxElement(node) ||
            isPropertyInAllowedCallExpression(node))) ||
        // Prevent reporting components created inside Array.map calls
        isMapCall(node) ||
        isMapCall(node.parent) ||
        // Do not mark components declared inside hooks (or falsy '() => null' clean-up methods)
        isReturnStatementOfHook(node, context) ||
        // Do not mark objects containing render methods
        isDirectValueOfRenderProperty(node) ||
        // Prevent reporting nested class components twice
        isInsideRenderMethod(node) ||
        // Prevent falsely reporting detected "components" which do not return JSX
        isStatelessComponentReturningNull(node)
      ) {
        return;
      }

      // Get the closest parent component
      const parentComponent = getClosestMatchingParent(node, context, (nodeToMatch) => components.get(nodeToMatch));

      if (parentComponent) {
        const parentName = resolveComponentName(parentComponent);

        // Exclude lowercase parents, e.g. function createTestComponent()
        // React-dom prevents creating lowercase components
        if (parentName && parentName[0] === parentName[0].toLowerCase()) {
          return;
        }

        const message = generateErrorMessageWithParentName(parentName);

        report(context, message, null, {
          node,
        });
      }
    }

    // --------------------------------------------------------------------------
    // Public
    // --------------------------------------------------------------------------

    return {
      FunctionDeclaration(node) {
        validate(node);
      },
      ArrowFunctionExpression(node) {
        validate(node);
      },
      FunctionExpression(node) {
        validate(node);
      },
      ClassDeclaration(node) {
        validate(node);
      },
      CallExpression(node) {
        validate(node);
      },
      TaggedTemplateExpression(node) {
        validate(node);
      },
    };
  }),
};
