/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { ModelListView, ModelListViewImpl } from './ModelListView';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import Utils from '../../common/utils/Utils';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';

const mockStore = configureStore([thunk, promiseMiddleware()]);

describe('ModelListView', () => {
  let wrapper;
  let minimalProps: any;
  let minimalStore: any;
  let onSearchSpy;
  beforeEach(() => {
    onSearchSpy = jest.fn();
    minimalProps = {
      models: [],
      searchInput: '',
      orderByKey: 'name',
      orderByAsc: true,
      currentPage: 1,
      nextPageToken: null,
      selectedStatusFilter: '',
      selectedOwnerFilter: '',
      onSearch: onSearchSpy,
      onClickNext: jest.fn(),
      onClickPrev: jest.fn(),
      onClickSortableColumn: jest.fn(),
      onSetMaxResult: jest.fn(),
      maxResultValue: 10,
      onStatusFilterChange: jest.fn(),
      onOwnerFilterChange: jest.fn(),
    };
    minimalStore = mockStore({});
  });
  function setupModelListViewWithIntl(propsParam: any) {
    const props = propsParam || minimalProps;
    return mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <ModelListView {...props} />
        </MemoryRouter>
      </Provider>,
    );
  }
  test('should render with minimal props without exploding', () => {
    // @ts-expect-error TS(2554): Expected 1 arguments, but got 0.
    wrapper = setupModelListViewWithIntl();
    expect(wrapper.length).toBe(1);
  });
  test('should not display onBoarding helper if disabled', () => {
    // @ts-expect-error TS(2554): Expected 1 arguments, but got 0.
    wrapper = setupModelListViewWithIntl();
    wrapper.find(ModelListViewImpl).setState({
      showOnboardingHelper: false,
    });
    expect(wrapper.find("[data-testid='showOnboardingHelper']").length).toBe(0);
  });
  test('Page title is set', () => {
    const mockUpdatePageTitle = jest.fn();
    Utils.updatePageTitle = mockUpdatePageTitle;
    // @ts-expect-error TS(2554): Expected 1 arguments, but got 0.
    wrapper = setupModelListViewWithIntl();
    expect(mockUpdatePageTitle.mock.calls[0][0]).toBe('MLflow Models');
  });
  // eslint-disable-next-line
});
