const { register } = require('@swc-node/register/register');
// If we don't pass any options to `register()`, swc-node will try to read the tsconfig from the
// current working directory (i.e. the directory we run lint from, which is different for every
// project being linted). We don't want it to do that, so I have to pass in at least one option to
// tell swc-node to use the given config instead of reading it from disk. Since the default
// transpilation options work fine, I'm passing a dummy option that has no effect in this context.
register({ strict: true });

module.exports = {
  rules: {
    'no-apollo-client-provider': require('./rules/no-apollo-client-provider'),
    'no-const-object-record-string': require('./rules/no-const-object-record-string'),
    'no-disable-lint': require('./rules/no-disable-lint').default,
    'no-direct-react-root': require('./rules/no-direct-react-root'),
    'no-direct-storage': require('./rules/no-direct-storage'),
    'no-dollar-signs-in-jsxtext': require('./rules/no-dollar-signs-in-jsxtext'),
    'no-double-negation': require('./rules/no-double-negation'),
    'no-dynamic-property-value': require('./rules/no-dynamic-property-value').default,
    'no-hardcoded-colors': require('./rules/no-hardcoded-colors'),
    'no-hardcoded-doc-links': require('./rules/no-hardcoded-doc-links'),
    'no-missing-react-hook-dependency-array': require('./rules/no-missing-react-hook-dependency-array'),
    'no-mock-location': require('./rules/no-mock-location'),
    'no-new-object-or-array-in-zustand-selector': require('./rules/no-new-object-or-array-in-zustand-selector'),
    'no-out-of-root-relative-imports': require('./rules/no-out-of-root-relative-imports'),
    'no-passive-modal-button-labels': require('./rules/no-passive-modal-button-labels').default,
    'no-react-prop-types': require('./rules/no-react-prop-types'),
    'no-restricted-globals-with-module': require('./rules/no-restricted-globals-with-module'),
    'no-restricted-imports-regexp': require('./rules/no-restricted-imports-regexp'),
    'no-singleton-query-client': require('./rules/no-singleton-query-client'),
    'no-restricted-jest-mock-modules': require('./rules/no-restricted-jest-mock-modules'),
    'no-uncaught-localstorage-setitem': require('./rules/no-uncaught-localstorage-setitem'),
    'no-unstable-nested-components': require('./rules/no-unstable-nested-components'),
    'no-use-react-table': require('./rules/no-use-react-table').default,
    'no-untyped-jest-require-actual': require('./rules/no-untyped-jest-require-actual'),
    'no-window-top': require('./rules/no-window-top').default,
    'no-wrapper-formui-label': require('./rules/no-wrapper-formui-label'),
    'react-lazy-only-at-top-level': require('./rules/react-lazy-only-at-top-level').default,
  },
};
