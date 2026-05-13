/**
 * This ESLint rule is designed to enforce best practices when subscribing to Zustand stores.
 * This helps in preventing unnecessary re-renders in React applications that utilize Zustand for state management.
 *
 * By default, Zustand uses referential equality to determine if a state change has occurred.
 * Returning a new object or array from a selector function will always be referentially different from the previous state.
 * This rule checks inline selectors and selectors defined as functions/variables within the same file.
 *
 * Correct usage:

    // Inline selector with shallow
    const { user, setUser } = useUserStore(state => ({
      user: state.user,
      setUser: state.setUser
    }), shallow);

    // Selector defined in the same file, used with shallow
    const selectData = state => ({ data: state.data });
    const { data } = useDataStore(selectData, shallow);

  OR

    // Select individual primitive slices
    const user = useUserStore(state => state.user);


 * Incorrect usage (Caught by this rule):

    // Inline selector missing shallow
    const { user, setUser } = useUserStore(state => ({
      user: state.user,
      setUser: state.setUser
    }));

    // Selector defined in the same file, used without shallow
    const selectData = state => ({ data: state.data }); // Defined in same file
    const { data } = useDataStore(selectData); // Missing shallow


 * Limitation: This rule CANNOT analyze selectors imported from different files.
 * If you import a selector, you must manually ensure `shallow` is used if that selector returns new objects/arrays.
 * Example (NOT CHECKED by this rule):
    import { selectUser } from './selectors'; // selectUser returns { user: state.user }
    const { user } = useUserStore(selectUser); // <-- Potential issue, but not flagged by this rule
 */
module.exports = {
  meta: {
    type: 'problem',
    docs: {
      description:
        "Disallow returning new objects or arrays in Zustand inline selectors or same-file function selectors without using Zustand's shallow equality function",
      category: 'Best Practices',
      recommended: false,
      url: 'https://github.com/pmndrs/zustand#selecting-multiple-state-slices',
    },
    schema: [], // No options
    messages: {
      noNewObjectOrArrayWithoutShallow:
        'When selecting data that results in a new object or array (from an inline function or a function defined in this file), use the `shallow` equality function from `zustand/shallow` as the equality comparer argument to prevent unnecessary re-renders. Alternatively, select primitive values individually.',
    },
  },
  create(context) {
    let shallowImportFound = false;

    function isFunctionNode(node) {
      return (
        node &&
        (node.type === 'ArrowFunctionExpression' ||
          node.type === 'FunctionExpression' ||
          node.type === 'FunctionDeclaration')
      );
    }

    // Checks if a function node's body returns a new object/array/function
    function doesFunctionNodeReturnNewObjectArrayOrFunction(funcNode) {
      if (!funcNode) return false;
      // Handle FunctionDeclaration directly
      const body = funcNode.body;
      if (!body) return false; // Should not happen for valid functions

      // Case 1: Arrow function with implicit return
      if (
        body.type === 'ObjectExpression' ||
        body.type === 'ArrayExpression' ||
        body.type === 'FunctionExpression' ||
        body.type === 'ArrowFunctionExpression'
      ) {
        return true;
      }

      // Case 2: Block statement (Arrow func body or FunctionDeclaration body)
      if (body.type === 'BlockStatement') {
        for (const statement of body.body) {
          if (statement.type === 'ReturnStatement' && statement.argument) {
            const returnArg = statement.argument;
            if (
              returnArg.type === 'ObjectExpression' ||
              returnArg.type === 'ArrayExpression' ||
              returnArg.type === 'FunctionExpression' ||
              returnArg.type === 'ArrowFunctionExpression'
            ) {
              return true;
            }
          }
        }
      }
      return false;
    }

    /**
     * Helper function to return the "selector" function depending on usage
     */
    function getResolvedNodeFunc(selectorArgNode) {
      let resolvedFuncNode = null; // The actual function node (inline or resolved from identifier)

      // Case 1: Selector argument is an inline function
      if (isFunctionNode(selectorArgNode)) {
        resolvedFuncNode = selectorArgNode;
      }
      // Case 2: Selector argument is an identifier - try to resolve in the same file
      else if (selectorArgNode.type === 'Identifier') {
        const identifierName = selectorArgNode.name;
        let scope = context.getScope(); // Get scope where the hook call occurs
        let variable;

        // Traverse up the scope chain to find where the identifier is defined
        // (handles cases where the selector is defined outside the immediate function scope, e.g., in module scope)
        while (scope) {
          variable = scope.variables.find((v) => v.name === identifierName);
          if (variable) {
            break; // Found it!
          }
          scope = scope.upper; // Move to the parent scope
        }

        if (variable && variable.defs.length > 0) {
          // Find the definition node (simplistic: use the last definition)
          // More robust check might involve looking at definition types or scope level
          const definition = variable.defs[variable.defs.length - 1];

          // Check if defined via VariableDeclarator (const/let/var selector = () => ...)
          if (
            definition.type === 'Variable' &&
            definition.node.type === 'VariableDeclarator' &&
            definition.node.init &&
            isFunctionNode(definition.node.init)
          ) {
            resolvedFuncNode = definition.node.init;
          }
          // Check if defined via FunctionDeclaration (function selector() ...)
          else if (
            definition.type === 'FunctionName' &&
            definition.node.type === 'FunctionDeclaration' &&
            isFunctionNode(definition.node)
          ) {
            // Check node itself is a function declaration
            resolvedFuncNode = definition.node;
          }
          // Add more checks here if needed (e.g., class methods)
        }
      }

      return resolvedFuncNode;
    }

    return {
      ImportDeclaration(node) {
        if (node.source.value === 'zustand/shallow') {
          node.specifiers.forEach((specifier) => {
            if (
              (specifier.type === 'ImportDefaultSpecifier' && specifier.local.name === 'shallow') ||
              (specifier.type === 'ImportSpecifier' &&
                specifier.imported.name === 'shallow' &&
                specifier.local.name === 'shallow')
            ) {
              shallowImportFound = true;
            }
          });
        }
      },

      CallExpression(node) {
        const callee = node.callee;
        const args = node.arguments;

        const isZustandHook =
          callee.type === 'Identifier' && (callee.name.match(/^use[A-Z].*Store$/) || callee.name === 'useStore');
        if (!isZustandHook) {
          return;
        }

        let selectorArgNode = null; // The argument node (inline func, identifier, etc.)
        let equalityFnNode = null;

        // Identify potential selector *argument* node and equality node based on signatures
        if (args.length >= 1) {
          // Default: Assume hook(selector, [equalityFn])
          selectorArgNode = args[0];
          equalityFnNode = args[1];

          // Refinement: If args[0] isn't a function-like thing, assume hook(store, selector, [equalityFn])
          // We check !isFunctionNode later, but this also handles non-identifiers safely
          if (args.length >= 2 && args[0].type !== 'ArrowFunctionExpression' && args[0].type !== 'FunctionExpression') {
            selectorArgNode = args[1];
            equalityFnNode = args[2];
          }
        }

        if (!selectorArgNode) {
          return; // No argument found in a potential selector position
        }

        const resolvedFuncNode = getResolvedNodeFunc(selectorArgNode);
        if (!resolvedFuncNode) {
          // Could not resolve the argument to a function node we can analyze.
          // This happens for imported selectors, complex assignments, non-functions, etc.
          // We do not report errors in these cases due to the analysis limitations.
          return;
        }

        // Now analyze the resolved function node
        const returnsNewObjectArrayOrFunction = doesFunctionNodeReturnNewObjectArrayOrFunction(resolvedFuncNode);

        if (!returnsNewObjectArrayOrFunction) {
          // The resolved selector function doesn't return a problematic value.
          return;
        }

        // Check if the 'shallow' equality function is provided correctly
        const isShallowUsedCorrectly =
          equalityFnNode && equalityFnNode.type === 'Identifier' && equalityFnNode.name === 'shallow';

        // Report error if a new object/array/function is returned WITHOUT proper shallow usage
        if (!isShallowUsedCorrectly || !shallowImportFound) {
          // Report the error on the original selector argument node (identifier or inline func)
          // This provides a more accurate location in the source code where the hook is called.
          context.report({
            node: selectorArgNode,
            messageId: 'noNewObjectOrArrayWithoutShallow',
          });
        }
      },
    };
  },
};
