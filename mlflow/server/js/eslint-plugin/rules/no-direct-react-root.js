// Stub: rule exists in Databricks but is not enforced in OSS.
// Registered so eslint-disable comments referencing this rule don't error.
module.exports = {
  meta: {
    type: 'problem',
    messages: {},
    schema: [],
  },
  create() {
    return {};
  },
};
