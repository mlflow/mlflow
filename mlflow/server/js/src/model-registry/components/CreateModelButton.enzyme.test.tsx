/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { CreateModelButton } from './CreateModelButton';
import { GenericInputModal } from '../../experiment-tracking/components/modals/GenericInputModal';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';

describe('CreateModelButton', () => {
  let wrapper: any;
  let minimalProps;
  let minimalStore: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    minimalProps = {};
    minimalStore = mockStore({});
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CreateModelButton {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    );
  });

  test('should render with minimal props and store without exploding', () => {
    expect(wrapper.find(CreateModelButton).length).toBe(1);
  });

  test('should render button type link correctly', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CreateModelButton buttonType="link" />
        </MemoryRouter>
      </Provider>,
    );
    expect(wrapper.find('.ant-btn-link').length).toBe(1);
  });

  test('should hide modal by default', () => {
    expect(wrapper.find(GenericInputModal).prop('isOpen')).toBe(false);
  });

  test('should show modal after button click', () => {
    wrapper.find('button.create-model-btn').simulate('click');
    expect(wrapper.find(GenericInputModal).prop('isOpen')).toBe(true);
  });
});
