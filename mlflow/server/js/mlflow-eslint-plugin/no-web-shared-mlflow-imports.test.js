// eslint-disable-next-line import/no-extraneous-dependencies
const { RuleTester } = require('eslint');
const rule = require('./no-web-shared-mlflow-imports');

const ruleTester = new RuleTester({
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module',
  },
});

const webSharedFilePath = '/repo/mlflow/server/js/src/shared/web-shared/example/file.ts';

ruleTester.run('no-web-shared-mlflow-imports', rule, {
  valid: [
    {
      filename: webSharedFilePath,
      code: `import { x } from './local-module';`,
    },
    {
      filename: webSharedFilePath,
      code: `import { x } from '../utils';`,
    },
    {
      filename: webSharedFilePath,
      code: `import { x } from '@databricks/web-shared/utils';`,
    },
    {
      filename: '/repo/mlflow/server/js/src/experiment-tracking/file.ts',
      code: `import { x } from '@mlflow/mlflow/src/common/utils/FetchUtils';`,
    },
  ],
  invalid: [
    {
      filename: webSharedFilePath,
      code: `import { x } from '@mlflow/mlflow/src/common/utils/FetchUtils';`,
      errors: [{ messageId: 'noMlflowImportsInWebShared' }],
    },
    {
      filename: webSharedFilePath,
      code: `import('@mlflow/mlflow/src/common/utils/FetchUtils');`,
      errors: [{ messageId: 'noMlflowImportsInWebShared' }],
    },
    {
      filename: webSharedFilePath,
      code: `export { x } from '../../../../experiment-tracking/routes';`,
      errors: [{ messageId: 'noMlflowImportsInWebShared' }],
    },
    {
      filename: webSharedFilePath,
      code: `import { x } from '../../../../common/utils/FetchUtils';`,
      errors: [{ messageId: 'noMlflowImportsInWebShared' }],
    },
  ],
});
