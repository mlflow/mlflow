import { setupServer } from '../../../common/utils/setup-msw';
import { NotFoundError } from '@databricks/web-shared/errors';
import { rest } from 'msw';
import { RegisteredPromptsApi } from './api';

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
