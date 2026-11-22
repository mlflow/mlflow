import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import { useEditEndpointForm, type TrafficSplitModel } from './useEditEndpointForm';
import { useEndpointQuery } from './useEndpointQuery';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useUpdateEndpointMutation } from './useUpdateEndpointMutation';
import { useCreateModelDefinitionMutation } from './useCreateModelDefinitionMutation';
import { useUpdateModelDefinitionMutation } from './useUpdateModelDefinitionMutation';
import { useCreateSecret } from './useCreateSecret';

jest.mock('../../common/utils/RoutingUtils', () => ({ useNavigate: () => jest.fn() }));
jest.mock('@mlflow/mlflow/src/common/utils/reactQueryHooks', () => ({
  useQueryClient: () => ({ invalidateQueries: jest.fn() }),
}));
jest.mock('./useEndpointsQuery');
jest.mock('./useEndpointQuery');
jest.mock('./useUpdateEndpointMutation');
jest.mock('./useCreateModelDefinitionMutation');
jest.mock('./useUpdateModelDefinitionMutation');
jest.mock('./useCreateSecret');

const mockMutation = () => ({ mutateAsync: jest.fn(), error: null, isLoading: false, reset: jest.fn() }) as any;

const buildModel = (apiKey: string): TrafficSplitModel => ({
  modelDefinitionName: 'model-def',
  provider: 'openai',
  modelName: 'gpt-4o',
  secretMode: 'new',
  existingSecretId: '',
  newSecret: { name: 'my-key', authMode: 'api_key', secretFields: { api_key: apiKey }, configFields: {} },
  weight: 100,
});

describe('useEditEndpointForm', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useEndpointsQuery).mockReturnValue({ data: [] } as any);
    jest.mocked(useEndpointQuery).mockReturnValue({ data: undefined, isLoading: false, error: null } as any);
    jest.mocked(useUpdateEndpointMutation).mockReturnValue(mockMutation());
    jest.mocked(useCreateModelDefinitionMutation).mockReturnValue(mockMutation());
    jest.mocked(useUpdateModelDefinitionMutation).mockReturnValue(mockMutation());
    jest.mocked(useCreateSecret).mockReturnValue(mockMutation());
  });

  test('isFormComplete is false when a new API key contains non-ASCII characters', () => {
    const { result } = renderHook(() => useEditEndpointForm('ep-1'));

    act(() => {
      result.current.form.setValue('trafficSplitModels', [buildModel('가나다')]);
    });
    expect(result.current.isFormComplete).toBe(false);

    act(() => {
      result.current.form.setValue('trafficSplitModels', [buildModel('rkskek')]);
    });
    expect(result.current.isFormComplete).toBe(true);
  });
});
