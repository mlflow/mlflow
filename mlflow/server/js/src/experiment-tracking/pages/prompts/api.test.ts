import { describe, it, expect } from '@jest/globals';
import { setupServer } from '../../../common/utils/setup-msw';

import { NotFoundError } from '@databricks/web-shared/errors';
import { rest } from 'msw';
import { RegisteredPromptsApi } from './api';
import { buildSearchFilterClause } from './utils';

describe('PromptsPage', () => {
  const server = setupServer();

  it('should properly return error when API responds with bare status', async () => {
    server.use(rest.get('/ajax-api/2.0/mlflow/registered-models/search', (req, res, ctx) => res(ctx.status(404))));

    const expectedMessage = new NotFoundError({}).message;

    await expect(RegisteredPromptsApi.listRegisteredPrompts()).rejects.toThrow(expectedMessage);
  });

  it('should properly return error with message extracted from API', async () => {
    server.use(
      rest.get('/ajax-api/2.0/mlflow/registered-models/search', (req, res, ctx) =>
        res(
          ctx.status(404),
          ctx.json({
            code: 'NOT_FOUND',
            message: 'Custom message: models not found',
          }),
        ),
      ),
    );

    await expect(RegisteredPromptsApi.listRegisteredPrompts()).rejects.toThrow('Custom message: models not found');
  });
});

describe('buildSearchFilterClause (re-exported from SearchUtils)', () => {
  it('re-exports the shared utility correctly', () => {
    expect(buildSearchFilterClause(undefined)).toBeUndefined();
    expect(buildSearchFilterClause('my-prompt')).toBe("name ILIKE '%my-prompt%'");
    expect(buildSearchFilterClause('name ILIKE "%test%"')).toBe('name ILIKE "%test%"');
  });
});

describe('RegisteredPromptsApi.listRegisteredPrompts with experimentId', () => {
  const server = setupServer();

  it('should include experiment ID filter when experimentId is provided', async () => {
    const experimentId = '123';
    let capturedFilter = '';

    server.use(
      rest.get('/ajax-api/2.0/mlflow/registered-models/search', (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter') || '';
        return res(
          ctx.json({
            registered_models: [],
          }),
        );
      }),
    );

    await RegisteredPromptsApi.listRegisteredPrompts(undefined, undefined, experimentId);

    expect(capturedFilter).toContain("tags.`mlflow.prompt.is_prompt` = 'true'");
    expect(capturedFilter).toContain(`tags.\`_mlflow_experiment_ids\` ILIKE '%,${experimentId},%'`);
    expect(capturedFilter).toContain(' AND ');
  });

  it('should combine experiment ID filter with search filter', async () => {
    const experimentId = '456';
    const searchFilter = 'my-prompt';
    let capturedFilter = '';

    server.use(
      rest.get('/ajax-api/2.0/mlflow/registered-models/search', (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter') || '';
        return res(
          ctx.json({
            registered_models: [],
          }),
        );
      }),
    );

    await RegisteredPromptsApi.listRegisteredPrompts(searchFilter, undefined, experimentId);

    expect(capturedFilter).toContain("tags.`mlflow.prompt.is_prompt` = 'true'");
    expect(capturedFilter).toContain(`tags.\`_mlflow_experiment_ids\` ILIKE '%,${experimentId},%'`);
    expect(capturedFilter).toContain("name ILIKE '%my-prompt%'");
  });

  it('should not include experiment ID filter when experimentId is not provided', async () => {
    let capturedFilter = '';

    server.use(
      rest.get('/ajax-api/2.0/mlflow/registered-models/search', (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter') || '';
        return res(
          ctx.json({
            registered_models: [],
          }),
        );
      }),
    );

    await RegisteredPromptsApi.listRegisteredPrompts();

    expect(capturedFilter).toContain("tags.`mlflow.prompt.is_prompt` = 'true'");
    expect(capturedFilter).not.toContain('_mlflow_experiment_ids');
  });
});

describe('RegisteredPromptsApi.createRegisteredPrompt with additionalTags', () => {
  const server = setupServer();

  it('should include additional tags when creating a prompt', async () => {
    let capturedBody: any = null;

    server.use(
      rest.post('/ajax-api/2.0/mlflow/registered-models/create', async (req, res, ctx) => {
        capturedBody = await req.json();
        return res(ctx.json({}));
      }),
    );

    const additionalTags = [{ key: '_mlflow_experiment_ids', value: ',789,' }];
    await RegisteredPromptsApi.createRegisteredPrompt('test-prompt', additionalTags);

    expect(capturedBody.name).toBe('test-prompt');
    expect(capturedBody.tags).toEqual(
      expect.arrayContaining([
        { key: 'mlflow.prompt.is_prompt', value: 'true' },
        { key: '_mlflow_experiment_ids', value: ',789,' },
      ]),
    );
  });

  it('should only include default tag when no additional tags provided', async () => {
    let capturedBody: any = null;

    server.use(
      rest.post('/ajax-api/2.0/mlflow/registered-models/create', async (req, res, ctx) => {
        capturedBody = await req.json();
        return res(ctx.json({}));
      }),
    );

    await RegisteredPromptsApi.createRegisteredPrompt('test-prompt');

    expect(capturedBody.name).toBe('test-prompt');
    expect(capturedBody.tags).toEqual([{ key: 'mlflow.prompt.is_prompt', value: 'true' }]);
  });
});
