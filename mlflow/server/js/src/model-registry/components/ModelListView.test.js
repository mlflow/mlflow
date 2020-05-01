import React from 'react';
import { shallow, mount } from 'enzyme';
import { ModelListView } from './ModelListView';
import { mockModelVersionDetailed, mockRegisteredModelDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { BrowserRouter } from 'react-router-dom';
import Utils from '../../common/utils/Utils';
import { ModelRegistryDocUrl } from '../../common/constants';

const ANTD_TABLE_PLACEHOLDER_CLS = '.ant-table-placeholder';

describe('ModelListView', () => {
  let wrapper;
  let instance;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      models: [],
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ModelListView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should show correct empty message', () => {
    wrapper = mount(<ModelListView {...minimalProps} />);
    expect(wrapper.find(`a[href="${ModelRegistryDocUrl}"]`)).toHaveLength(1);

    instance = wrapper.instance();
    instance.setState({ nameFilter: 'xyz' });
    expect(wrapper.find(ANTD_TABLE_PLACEHOLDER_CLS).text()).toBe('No models found.');
  });

  test('should render latest version correctly', () => {
    const models = [
      mockRegisteredModelDetailed('Model A', [
        mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY),
        mockModelVersionDetailed('Model A', 2, Stages.STAGING, ModelVersionStatus.READY),
        mockModelVersionDetailed('Model A', 3, Stages.NONE, ModelVersionStatus.READY),
      ]),
    ];
    const props = { ...minimalProps, models };
    wrapper = mount(
      <BrowserRouter>
        <ModelListView {...props} />
      </BrowserRouter>,
    );
    expect(wrapper.find('td.latest-version').text()).toBe('Version 3');
    expect(wrapper.find('td.latest-staging').text()).toBe('Version 2');
    expect(wrapper.find('td.latest-production').text()).toBe('Version 1');
  });

  test('should render `_` when there is no version to display for the cell', () => {
    const models = [
      mockRegisteredModelDetailed('Model A', [
        mockModelVersionDetailed('Model A', 1, Stages.NONE, ModelVersionStatus.READY),
      ]),
    ];
    const props = { ...minimalProps, models };
    wrapper = mount(
      <BrowserRouter>
        <ModelListView {...props} />
      </BrowserRouter>,
    );
    expect(wrapper.find('td.latest-version').text()).toBe('Version 1');
    expect(wrapper.find('td.latest-staging').text()).toBe('_');
    expect(wrapper.find('td.latest-production').text()).toBe('_');
  });

  test('should apply name based search correctly', () => {
    const models = [
      mockRegisteredModelDetailed('Model A', []),
      mockRegisteredModelDetailed('Model B', []),
    ];
    const props = { ...minimalProps, models };
    wrapper = mount(
      <BrowserRouter>
        <ModelListView {...props} />
      </BrowserRouter>,
    );
    expect(wrapper.find('td.model-name').length).toBe(2);
    expect(
      wrapper
        .find('td.model-name')
        .first()
        .text(),
    ).toBe('Model A');
    expect(
      wrapper
        .find('td.model-name')
        .last()
        .text(),
    ).toBe('Model B');

    instance = wrapper.find(ModelListView).instance();
    instance.setState({ nameFilter: 'a' }); // apply name search 'a'
    wrapper.update(); // For some reason we need a force update to catch up here
    expect(wrapper.find('td.model-name').length).toBe(1);
    expect(wrapper.find('td.model-name').text()).toBe('Model A');
  });

  test('should by default sort by model name alphabetically and case insensitively', () => {
    // Intentionally shuffled by model names
    const models = [
      mockRegisteredModelDetailed('Model B', []),
      mockRegisteredModelDetailed('model c', []),
      mockRegisteredModelDetailed('Model a', []),
    ];
    const props = { ...minimalProps, models };
    wrapper = mount(
      <BrowserRouter>
        <ModelListView {...props} />
      </BrowserRouter>,
    );
    expect(wrapper.find('td.model-name').length).toBe(3);
    expect(
      wrapper
        .find('td.model-name')
        .at(0)
        .text(),
    ).toBe('Model a');
    expect(
      wrapper
        .find('td.model-name')
        .at(1)
        .text(),
    ).toBe('Model B');
    expect(
      wrapper
        .find('td.model-name')
        .at(2)
        .text(),
    ).toBe('model c');
  });

  test('Page title is set', () => {
    const mockUpdatePageTitle = jest.fn();
    Utils.updatePageTitle = mockUpdatePageTitle;
    wrapper = shallow(<ModelListView {...minimalProps} />);
    expect(mockUpdatePageTitle.mock.calls[0][0]).toBe('MLflow Models');
  });
});
