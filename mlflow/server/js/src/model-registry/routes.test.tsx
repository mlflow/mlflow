import { ModelRegistryRoutes } from './routes';

describe('model registry page routes working in Databricks path-based router', () => {
  test('yields correct route paths for listing page', () => {
    expect(ModelRegistryRoutes.modelListPageRoute).toEqual('/models');
  });
});
