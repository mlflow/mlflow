import { describe, it, expect } from '@jest/globals';
import { setupServer } from '../common/utils/setup-msw';
import { NotFoundError } from '@databricks/web-shared/errors';
import { rest } from 'msw';
import { GatewayApi } from './api';

describe('GatewayApi', () => {
  const server = setupServer();

  describe('Provider Metadata', () => {
    it('should properly return error when listProviders API responds with bare status', async () => {
      server.use(rest.get('/ajax-api/3.0/mlflow/gateway/supported-providers', (req, res, ctx) => res(ctx.status(404))));

      const expectedMessage = new NotFoundError({}).message;

      await expect(GatewayApi.listProviders()).rejects.toThrow(expectedMessage);
    });

    it('should properly return error with message extracted from listModels API', async () => {
      server.use(
        rest.get('/ajax-api/3.0/mlflow/gateway/supported-models', (req, res, ctx) =>
          res(
            ctx.status(500),
            ctx.json({
              code: 'INTERNAL_ERROR',
              message: 'Failed to fetch models',
            }),
          ),
        ),
      );

      await expect(GatewayApi.listModels()).rejects.toThrow('Failed to fetch models');
    });
  });

  describe('Secrets Management', () => {
    it('should properly return error when createSecret API fails', async () => {
      server.use(
        rest.post('/ajax-api/3.0/mlflow/gateway/secrets/create', (req, res, ctx) =>
          res(
            ctx.status(400),
            ctx.json({
              code: 'INVALID_PARAMETER_VALUE',
              message: 'Secret name already exists',
            }),
          ),
        ),
      );

      await expect(
        GatewayApi.createSecret({
          secret_name: 'duplicate-secret',
          secret_value: { api_key: 'test-value' },
        }),
      ).rejects.toThrow('Secret name already exists');
    });

    it('should properly return error when listSecrets API responds with bare status', async () => {
      server.use(rest.get('/ajax-api/3.0/mlflow/gateway/secrets/list', (req, res, ctx) => res(ctx.status(403))));

      await expect(GatewayApi.listSecrets()).rejects.toThrow();
    });
  });

  describe('Endpoints Management', () => {
    it('should properly return error when createEndpoint API fails', async () => {
      server.use(
        rest.post('/ajax-api/3.0/mlflow/gateway/endpoints/create', (req, res, ctx) =>
          res(
            ctx.status(400),
            ctx.json({
              code: 'INVALID_PARAMETER_VALUE',
              message: 'At least one model definition is required',
            }),
          ),
        ),
      );

      await expect(
        GatewayApi.createEndpoint({
          name: 'test-endpoint',
          model_configs: [],
        }),
      ).rejects.toThrow('At least one model definition is required');
    });

    it('should properly return error when listEndpoints API responds with bare status', async () => {
      server.use(rest.get('/ajax-api/3.0/mlflow/gateway/endpoints/list', (req, res, ctx) => res(ctx.status(500))));

      await expect(GatewayApi.listEndpoints()).rejects.toThrow();
    });
  });

  describe('Model Definitions Management', () => {
    it('should properly return error when createModelDefinition API fails', async () => {
      server.use(
        rest.post('/ajax-api/3.0/mlflow/gateway/model-definitions/create', (req, res, ctx) =>
          res(
            ctx.status(400),
            ctx.json({
              code: 'INVALID_PARAMETER_VALUE',
              message: 'Model definition name already exists',
            }),
          ),
        ),
      );

      await expect(
        GatewayApi.createModelDefinition({
          name: 'duplicate-model',
          secret_id: 'secret-123',
          provider: 'openai',
          model_name: 'gpt-4',
        }),
      ).rejects.toThrow('Model definition name already exists');
    });

    it('should properly return error when listModelDefinitions API responds with bare status', async () => {
      server.use(
        rest.get('/ajax-api/3.0/mlflow/gateway/model-definitions/list', (req, res, ctx) => res(ctx.status(500))),
      );

      await expect(GatewayApi.listModelDefinitions()).rejects.toThrow();
    });
  });
});
