import { shallow } from 'enzyme';
import LocalStorageUtils from '../../../../common/utils/LocalStorageUtils';
import { useExperimentViewLocalStore } from './useExperimentViewLocalStore';

jest.mock('../../../../common/utils/LocalStorageUtils');

describe('useExperimentViewLocalStore', () => {
  it('tests useExperimentViewLocalStore', () => {
    const Component = () => {
      useExperimentViewLocalStore('123');
      return null;
    };

    shallow(<Component />);
    expect(LocalStorageUtils.getStoreForComponent).toBeCalledWith('ExperimentView', '123');
  });
});
