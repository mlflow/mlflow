import { shouldUsePathRouting } from '../common/utils/FeatureUtils';
import { ModelRegistryRoutes } from './routes';

jest.mock('../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../common/utils/FeatureUtils'),
  shouldUsePathRouting: jest.fn(),
}));

describe('model registry page routes', () => {
  beforeEach(() => {
    (shouldUsePathRouting as jest.Mock).mockReturnValue(false);
  });
  test('yields correct route paths for listing page', () => {
    expect(ModelRegistryRoutes.modelListPageRoute).toEqual('/models');
  });

  test('yields correct route paths for model page routes', () => {
    expect(ModelRegistryRoutes.getModelPageRoute('model-name')).toEqual('/models/model-name');
    expect(ModelRegistryRoutes.getModelPageServingRoute('model-name')).toEqual(
      '/models/model-name/serving',
    );
  });
  test('yields correct route paths for model version page routes', () => {
    expect(ModelRegistryRoutes.getModelVersionPageRoute('model-name', '6')).toEqual(
      '/models/model-name/versions/6',
    );
    expect(
      ModelRegistryRoutes.getCompareModelVersionsPageRoute('model-name', {
        'run-1': '1',
        'run-2': '2',
        'run-3': '3',
      }),
    ).toEqual(
      '/compare-model-versions?name="model-name"&runs={"run-1":"1","run-2":"2","run-3":"3"}',
    );
  });
});
