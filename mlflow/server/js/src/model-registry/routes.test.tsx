import { describe, expect, test } from '@jest/globals';
import { ModelRegistryRoutes } from './routes';

describe('model registry page routes working in Databricks path-based router', () => {
  test('yields correct route paths for listing page', () => {
    // eslint-disable-next-line jest/no-standalone-expect
    expect(ModelRegistryRoutes.modelListPageRoute).toEqual('/models');
  });
});
