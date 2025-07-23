import { renderHook } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import LocalStorageUtils from '../../../../common/utils/LocalStorageUtils';
import { useExperimentViewLocalStore } from './useExperimentViewLocalStore';

jest.mock('../../../../common/utils/LocalStorageUtils');

describe('useExperimentViewLocalStore', () => {
  it('tests useExperimentViewLocalStore', () => {
    renderHook(() => useExperimentViewLocalStore('123'));
    expect(LocalStorageUtils.getStoreForComponent).toHaveBeenCalledWith('ExperimentView', '123');
  });
});
