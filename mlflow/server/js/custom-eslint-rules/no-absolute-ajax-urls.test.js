/**
 * Tests for no-absolute-ajax-urls custom ESLint rule
 */

const { RuleTester } = require('eslint');
const rule = require('./no-absolute-ajax-urls');

const ruleTester = new RuleTester({
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module',
  },
});

ruleTester.run('no-absolute-ajax-urls', rule, {
  valid: [
    {
      code: `const url = getAjaxUrl('ajax-api/2.0/mlflow/experiments');`,
    },
    // URLs that don't contain '/ajax-api/'
    {
      code: `const url = '/api/v1/users';`,
    },
    {
      code: `const url = 'https://example.com/api';`,
    },
    {
      code: `const url = \`/some/other/\${path}\`;`,
    },
    // comments mentioning '/ajax-api/' should be allowed
    {
      code: `// This uses /ajax-api/ endpoint\nconst url = 'ajax-api/2.0/mlflow';`,
    },
  ],

  invalid: [
    // string literal with leading slash
    {
      code: `const url = '/ajax-api/2.0/mlflow/experiments';`,
      errors: [
        {
          messageId: 'absoluteAjaxUrl',
        },
      ],
    },
    // fetch call with absolute URL
    {
      code: `fetch('/ajax-api/2.0/mlflow/runs');`,
      errors: [
        {
          messageId: 'absoluteAjaxUrl',
        },
      ],
    },
    // fetch call with absolute URL
    {
      code: `fetch('/ajax-api/2.0/mlflow/experiments/list');`,
      errors: [
        {
          messageId: 'absoluteAjaxUrl',
        },
      ],
    },
    // template literal with leading slash
    {
      code: `const url = \`/ajax-api/2.0/mlflow/experiments/\${id}\`;`,
      errors: [
        {
          messageId: 'absoluteAjaxUrl',
        },
      ],
    },
    // within object
    {
      code: `const config = { url: '/ajax-api/2.0/mlflow/experiments' };`,
      errors: [
        {
          messageId: 'absoluteAjaxUrl',
        },
      ],
    },
  ],
});
