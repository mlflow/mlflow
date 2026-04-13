/**
 * Tests for componentId-must-be-in-registry custom ESLint rule
 */
// eslint-disable-next-line import/no-extraneous-dependencies
const { RuleTester } = require('eslint');

// Mock the registry for testing — override the require cache
const registryPath = require.resolve('./componentId-registry');
const originalRegistry = require(registryPath);

// Install a test registry with known entries
require.cache[registryPath] = {
  id: registryPath,
  filename: registryPath,
  loaded: true,
  exports: {
    'mlflow.test.registered': 'A registered test componentId',
    'mlflow.test.another': 'Another registered componentId',
    'codegen_test_123': 'A codegen componentId',
  },
};

// Re-require the rule so it picks up the mocked registry
delete require.cache[require.resolve('./componentId-must-be-in-registry')];
const rule = require('./componentId-must-be-in-registry');

const ruleTester = new RuleTester({
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module',
    ecmaFeatures: { jsx: true },
  },
});

ruleTester.run('componentId-must-be-in-registry', rule, {
  valid: [
    // JSX prop with registered ID
    {
      code: `<Button componentId="mlflow.test.registered" />`,
    },
    // Object property with registered ID
    {
      code: `const x = { componentId: "mlflow.test.another" };`,
    },
    // data-component-id with registered ID
    {
      code: `<button data-component-id="mlflow.test.registered" />`,
    },
    // Codegen ID in registry
    {
      code: `<Tag componentId="codegen_test_123" />`,
    },
    // Dynamic template literal — should be skipped (not checked)
    {
      code: `<Button componentId={\`\${prefix}.submit\`} />`,
    },
    // Variable reference — should be skipped
    {
      code: `<Button componentId={someVar} />`,
    },
    // String not used as componentId — not checked
    {
      code: `const x = "not.in.registry";`,
    },
    // componentId in JSX expression container with registered ID
    {
      code: `<Button componentId={"mlflow.test.registered"} />`,
    },
    // Ternary with both branches registered
    {
      code: `<Button componentId={cond ? "mlflow.test.registered" : "mlflow.test.another"} />`,
    },
  ],

  invalid: [
    // Unregistered JSX componentId
    {
      code: `<Button componentId="not.in.registry" />`,
      errors: [{ messageId: 'unregisteredComponentId' }],
    },
    // Unregistered object property componentId
    {
      code: `const x = { componentId: "also.not.registered" };`,
      errors: [{ messageId: 'unregisteredComponentId' }],
    },
    // Unregistered data-component-id
    {
      code: `<select data-component-id="not.registered" />`,
      errors: [{ messageId: 'unregisteredComponentId' }],
    },
    // Unregistered JSX expression container
    {
      code: `<Button componentId={"unknown.id"} />`,
      errors: [{ messageId: 'unregisteredComponentId' }],
    },
    // Ternary with one unregistered branch
    {
      code: `<Button componentId={cond ? "mlflow.test.registered" : "not.registered"} />`,
      errors: [{ messageId: 'unregisteredComponentId' }],
    },
  ],
});

// Restore original registry
require.cache[registryPath].exports = originalRegistry;

console.log('All componentId-must-be-in-registry tests passed!');
