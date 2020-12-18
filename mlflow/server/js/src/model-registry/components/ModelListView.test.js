import React from 'react';
import { shallow, mount } from 'enzyme';
import { ModelListView } from './ModelListView';
import { mockModelVersionDetailed, mockRegisteredModelDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { BrowserRouter } from 'react-router-dom';
import Utils from '../../common/utils/Utils';
import { ModelRegistryDocUrl } from '../../common/constants';
import { Table, Input } from 'antd';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';

const mockStore = configureStore([thunk, promiseMiddleware()]);

const { Search } = Input;

const ANTD_TABLE_PLACEHOLDER_CLS = '.ant-table-placeholder';

describe('ModelListView', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let minimalStore;

  beforeEach(() => {
    minimalProps = {
      models: [],
      searchInput: '',
      orderByKey: 'name',
      orderByAsc: true,
      currentPage: 1,
      nextPageToken: null, // no next page
      onSearch: jest.fn(),
      onClickNext: jest.fn(),
      onClickPrev: jest.fn(),
      onClickSortableColumn: jest.fn(),
      onSetMaxResult: jest.fn(),
      getMaxResultValue: jest.fn().mockReturnValue(10),
    };
    minimalStore = mockStore({});
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ModelListView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should display onBoarding helper', () => {
    wrapper = shallow(<ModelListView {...minimalProps} />);
    expect(wrapper.find('Alert').length).toBe(1);
  });

  test('should not display onBoarding helper if disabled', () => {
    wrapper = shallow(<ModelListView {...minimalProps} />);
    instance = wrapper.instance();
    instance.setState({
      showOnboardingHelper: false,
    });
    expect(wrapper.find('Alert').length).toBe(0);
  });

  test('should show correct link in onboarding helper', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(`a[href="${ModelRegistryDocUrl}"]`)).toHaveLength(1);
  });

  test('should render correct information if table is empty', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(ANTD_TABLE_PLACEHOLDER_CLS).text()).toBe(
      'No models yet. Create a model to get started.',
    );

    wrapper.setProps({
      children: (
        <BrowserRouter>
          <ModelListView {...{ ...minimalProps, searchInput: 'xyz' }} />
        </BrowserRouter>
      ),
    });
    expect(wrapper.find(ANTD_TABLE_PLACEHOLDER_CLS).text()).toBe('No models found.');

    instance = wrapper.find(ModelListView).instance();
    instance.setState({ lastNavigationActionWasClickPrev: true });
    expect(wrapper.find(ANTD_TABLE_PLACEHOLDER_CLS).text()).toBe(
      'No models found for the page. ' +
        'Please refresh the page as the underlying data may have changed significantly.',
    );
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
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListView {...props} />
        </BrowserRouter>
      </Provider>,
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
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListView {...props} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find('td.latest-version').text()).toBe('Version 1');
    expect(wrapper.find('td.latest-staging').text()).toBe('_');
    expect(wrapper.find('td.latest-production').text()).toBe('_');
  });

  test('the search input is called with prop searchInput value', () => {
    wrapper = shallow(<ModelListView {...minimalProps} />);
    expect(wrapper.find(Search).props().defaultValue).toBe('');

    wrapper.setProps({ searchInput: 'xyz' });
    expect(wrapper.find(Search).props().defaultValue).toBe('xyz');
  });

  const findColumn = (table, index) =>
    table.props().columns.find((elem) => elem.dataIndex === index);

  test('orderByKey, orderByASC props are correctly passed to the table', () => {
    const models = [
      mockRegisteredModelDetailed('Model B', [], [], 'CAN_EDIT', 3),
      mockRegisteredModelDetailed('model c', [], [], 'CAN_EDIT', 1),
      mockRegisteredModelDetailed('Model a', [], [], 'CAN_EDIT', 2),
    ];
    let props = {
      ...minimalProps,
      models,
      orderByKey: 'name',
      orderByAsc: true,
    };
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListView {...props} />
        </BrowserRouter>
      </Provider>,
    );

    let table = wrapper.find(Table);
    // prop values look legit
    expect(findColumn(table, 'name').sortOrder).toBe('ascend');
    expect(findColumn(table, 'last_updated_timestamp').sortOrder).toBe(undefined);
    // the table doesn't actually sort, though, and displays exactly what's given.
    expect(wrapper.find('td.model-name').length).toBe(3);
    expect(
      wrapper
        .find('td.model-name')
        .at(0)
        .text(),
    ).toBe('Model B');
    expect(
      wrapper
        .find('td.model-name')
        .at(1)
        .text(),
    ).toBe('model c');
    expect(
      wrapper
        .find('td.model-name')
        .at(2)
        .text(),
    ).toBe('Model a');

    props = {
      ...minimalProps,
      models,
      orderByKey: 'timestamp',
      orderByAsc: false,
    };
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListView {...props} />
        </BrowserRouter>
      </Provider>,
    );
    table = wrapper.find(Table);
    // prop values look legit
    expect(findColumn(table, 'name').sortOrder).toBe(undefined);
    expect(findColumn(table, 'last_updated_timestamp').sortOrder).toBe('descend');
    // the table doesn't actually sort, though, and displays exactly what's given.
    expect(wrapper.find('td.model-name').length).toBe(3);
    expect(
      wrapper
        .find('td.model-name')
        .at(0)
        .text(),
    ).toBe('Model B');
    expect(
      wrapper
        .find('td.model-name')
        .at(1)
        .text(),
    ).toBe('model c');
    expect(
      wrapper
        .find('td.model-name')
        .at(2)
        .text(),
    ).toBe('Model a');
  });

  test('lastNavigationActionWasClickPrev is set properly on actions', () => {
    wrapper = shallow(<ModelListView {...minimalProps} />);
    instance = wrapper.instance();
    expect(instance.state.lastNavigationActionWasClickPrev).toBe(false);

    instance.handleClickPrev();
    expect(instance.state.lastNavigationActionWasClickPrev).toBe(true);
    instance.handleClickNext();
    expect(instance.state.lastNavigationActionWasClickPrev).toBe(false);
    instance.handleSearch('');
    expect(instance.state.lastNavigationActionWasClickPrev).toBe(false);
    instance.handleTableChange(null, null, { field: 'name', order: 'ascend' });
    expect(instance.state.lastNavigationActionWasClickPrev).toBe(false);
  });

  test('Page title is set', () => {
    const mockUpdatePageTitle = jest.fn();
    Utils.updatePageTitle = mockUpdatePageTitle;
    wrapper = shallow(<ModelListView {...minimalProps} />);
    expect(mockUpdatePageTitle.mock.calls[0][0]).toBe('MLflow Models');
  });
});
