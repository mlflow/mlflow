import React from 'react';
import { shallow, mount } from 'enzyme';
import { ModelVersionTable } from './ModelVersionTable';
import { mockModelVersionDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { Table } from 'antd';
import { RegisteringModelDocUrl } from '../../common/constants';

describe('ModelVersionTable', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      modelName: 'Model A',
      modelVersions: [],
      onChange: jest.fn(),
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ModelVersionTable {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render correct empty text', () => {
    wrapper = mount(<ModelVersionTable {...minimalProps} />);
    expect(wrapper.find(`a[href="${RegisteringModelDocUrl}"]`)).toHaveLength(1);
  });

  test('should render active versions when activeStageOnly is true', () => {
    const props = {
      ...minimalProps,
      modelVersions: [
        mockModelVersionDetailed('Model A', 1, Stages.NONE, ModelVersionStatus.READY),
        mockModelVersionDetailed('Model A', 2, Stages.PRODUCTION, ModelVersionStatus.READY),
        mockModelVersionDetailed('Model A', 3, Stages.STAGING, ModelVersionStatus.READY),
        mockModelVersionDetailed('Model A', 4, Stages.ARCHIVED, ModelVersionStatus.READY),
      ],
    };
    wrapper = shallow(<ModelVersionTable {...props} />);
    expect(wrapper.find(Table).props().dataSource.length).toBe(4);
    wrapper.setProps({ activeStageOnly: true });
    expect(wrapper.find(Table).props().dataSource.length).toBe(2);
  });
});
