import React from 'react';
import { ModelVersionTable } from './ModelVersionTable';
import { mockModelVersionDetailed, stageTagComponents, modelStageNames } from '../test-utils';
import { ModelVersionStatus } from '../constants';
import { Table } from 'antd';
import { RegisteringModelDocUrl } from '../../common/constants';
import { mountWithIntl } from '../../common/utils/TestUtils';
import { Stages } from '../test-utils';
import { BrowserRouter } from 'react-router-dom';

describe('ModelVersionTable', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      modelName: 'Model A',
      modelVersions: [],
      onChange: jest.fn(),
      stageTagComponents: stageTagComponents(),
      allStagesAvailable: modelStageNames
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <BrowserRouter>
        <ModelVersionTable {...minimalProps} />
      </BrowserRouter>,
    );
    expect(wrapper.length).toBe(1);
  });

  test('should render correct empty text', () => {
    wrapper = wrapper = mountWithIntl(
      <BrowserRouter>
        <ModelVersionTable {...minimalProps} />
      </BrowserRouter>,
    );
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
    wrapper = mountWithIntl(
      <BrowserRouter>
        <ModelVersionTable {...props} />
      </BrowserRouter>,
    );
    expect(wrapper.find(Table).props().dataSource.length).toBe(4);

    const propsWithActive = {
      ...props,
      activeStageOnly: true,
    };
    wrapper = mountWithIntl(
      <BrowserRouter>
        <ModelVersionTable {...propsWithActive} />
      </BrowserRouter>,
    );
    expect(wrapper.find(Table).props().dataSource.length).toBe(2);
  });
});
