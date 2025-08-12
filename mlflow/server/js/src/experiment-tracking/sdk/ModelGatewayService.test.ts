import { ModelGatewayService } from './ModelGatewayService';
import { ModelGatewayRouteTask } from './MlflowEnums';
import { fetchEndpoint } from '../../common/utils/FetchUtils';
import { MlflowService } from './MlflowService';

jest.mock('../../common/utils/FetchUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/FetchUtils')>('../../common/utils/FetchUtils'),
  fetchEndpoint: jest.fn(),
}));

describe('ModelGatewayService', () => {
  beforeEach(() => {
    jest
      .spyOn(MlflowService, 'gatewayProxyPost')
      .mockResolvedValue({ choices: [{ message: { content: 'output text' } }], usage: {} });
  });
  afterEach(() => {
    jest.clearAllMocks();
  });

  test('Creates a request call to the MLflow deployments model route', async () => {
    const result = await ModelGatewayService.queryModelGatewayRoute(
      {
        name: 'chat_route',
        key: 'mlflow_deployment_endpoint:test-mlflow-deployment-endpoint-chat',
        task: ModelGatewayRouteTask.LLM_V1_CHAT,
        type: 'mlflow_deployment_endpoint',
        mlflowDeployment: {
          endpoint_type: ModelGatewayRouteTask.LLM_V1_CHAT,
          endpoint_url: '/endpoint-url',
          model: {
            name: 'mpt-7b',
            provider: 'mosaic',
          },
          name: 'test-mlflow-deployment-endpoint-chat',
        },
      },
      { inputText: 'input text', parameters: { temperature: 0.5, max_tokens: 50 } },
    );

    expect(MlflowService.gatewayProxyPost).toHaveBeenCalledWith(
      expect.objectContaining({
        gateway_path: 'endpoint-url',
        json_data: { messages: [{ content: 'input text', role: 'user' }], temperature: 0.5, max_tokens: 50 },
      }),
    );

    expect(result).toEqual(
      expect.objectContaining({
        text: 'output text',
      }),
    );
  });

  test('Throws when task is not supported', async () => {
    try {
      await ModelGatewayService.queryModelGatewayRoute(
        {
          name: 'embeddings_route',
          key: 'mlflow_deployment_endpoint:test-mlflow-deployment-endpoint-embeddings',
          task: ModelGatewayRouteTask.LLM_V1_EMBEDDINGS,
          type: 'mlflow_deployment_endpoint',
          mlflowDeployment: {
            endpoint_type: ModelGatewayRouteTask.LLM_V1_EMBEDDINGS,
            endpoint_url: '/endpoint-url',
            model: {
              name: 'mpt-7b',
              provider: 'mosaic',
            },
            name: 'test-mlflow-deployment-endpoint-embeddings',
          },
        },
        { inputText: 'input text', parameters: { temperature: 0.5, max_tokens: 50 } },
      );
    } catch (e: any) {
      expect(e.message).toMatch(/Unsupported served LLM model task/);
    }
  });

  test('Throws when route type is not supported', async () => {
    try {
      await ModelGatewayService.queryModelGatewayRoute(
        {
          type: 'some-unsupported-type',
          name: 'completions_route',
        } as any,
        { inputText: 'input text', parameters: { temperature: 0.5, max_tokens: 50 } },
      );
    } catch (e: any) {
      expect(e.message).toMatch(/Unknown route type/);
    }
  });
});
