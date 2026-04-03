const { register } = require('@swc-node/register/register');
// If we don't pass any options to `register()`, swc-node will try to read the tsconfig from the
// current working directory (i.e. the directory we run lint from, which is different for every
// project being linted). We don't want it to do that, so I have to pass in at least one option to
// tell swc-node to use the given config instead of reading it from disk. Since the default
// transpilation options work fine, I'm passing a dummy option that has no effect in this context.
register({ strict: true });

module.exports = {
  rules: {
    'no-dynamic-property-value': require('./rules/no-dynamic-property-value').default,
  },
};
