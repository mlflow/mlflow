/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { ModelVersionTable } from './ModelVersionTable';
import { mockModelVersionDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { Table } from 'antd';
import { RegisteringModelDocUrl } from '../../common/constants';
import { mountWithIntl } from '../../common/utils/TestUtils';
import { MemoryRouter } from 'react-router-dom-v5-compat';
describe('ModelVersionTable', () => {
  let wrapper;
  let minimalProps: any;
  beforeEach(() => {
    minimalProps = {
      modelName: 'Model A',
      modelVersions: [],
      onChange: jest.fn(),
    };
  });
  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <MemoryRouter>
        <ModelVersionTable {...minimalProps} />
      </MemoryRouter>,
    );
    expect(wrapper.length).toBe(1);
  });
  test('should render correct empty text', () => {
    wrapper = wrapper = mountWithIntl(
      <MemoryRouter>
        <ModelVersionTable {...minimalProps} />
      </MemoryRouter>,
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
      <MemoryRouter>
        <ModelVersionTable {...props} />
      </MemoryRouter>,
    );
    expect(wrapper.find(Table).props().dataSource.length).toBe(4);
    const propsWithActive = {
      ...props,
      activeStageOnly: true,
    };
    wrapper = mountWithIntl(
      <MemoryRouter>
        <ModelVersionTable {...propsWithActive} />
      </MemoryRouter>,
    );
    expect(wrapper.find(Table).props().dataSource.length).toBe(2);
  });
});
