module.exports = {
  meta: {
    type: 'problem',
    docs: {
      description:
        'Enforce using useLocalStorage()/useSessionStorage() hooks instead of directly accessing window.localStorage/sessionStorage or their globals',
      recommended: true,
    },
    messages: {
      disallowLocalStorage:
        'Avoid direct localStorage access. ' +
        'Use a `useLocalStorage()` hook instead, which provides scoped key management.',
      disallowSessionStorage:
        'Avoid direct sessionStorage access. ' +
        'Use a `useSessionStorage()` hook instead, which provides scoped key management.',
    },
  },
  create(context) {
    /**
     * Checks if the given node is accessing a storage global (localStorage or sessionStorage)
     * and reports if so.
     * @param {object} node - The MemberExpression AST node
     * @param {string} storageName - Either 'localStorage' or 'sessionStorage'
     * @param {string} messageId - The message ID to report
     */
    function checkStorageAccess(node, storageName, messageId) {
      // Pattern 1: window.localStorage.* or window.sessionStorage.*
      const isWindowStorage =
        node.object.type === 'MemberExpression' &&
        node.object.object.name === 'window' &&
        node.object.property.name === storageName;

      // Pattern 2: bare localStorage.* or sessionStorage.*
      const isBareStorage = node.object.type === 'Identifier' && node.object.name === storageName;

      if (isWindowStorage || isBareStorage) {
        // For bare storage, check if it's a local variable or the global
        if (isBareStorage) {
          const scope = context.sourceCode.getScope ? context.sourceCode.getScope(node) : context.getScope();
          const variable = scope.set.get(storageName);

          // If storage is defined as a local variable, don't report it
          if (variable && variable.defs.length > 0) {
            return;
          }
        }

        context.report({ node, messageId });
      }
    }

    return {
      MemberExpression(node) {
        checkStorageAccess(node, 'localStorage', 'disallowLocalStorage');
        checkStorageAccess(node, 'sessionStorage', 'disallowSessionStorage');
      },
    };
  },
};
