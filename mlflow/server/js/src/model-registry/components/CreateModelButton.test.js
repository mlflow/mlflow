import React from 'react';
import { mount } from 'enzyme';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { CreateModelButton } from './CreateModelButton';
import { GenericInputModal } from '../../experiment-tracking/components/modals/GenericInputModal';

describe('CreateModelButton', () => {
  let wrapper;
  let minimalProps;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    minimalProps = {};
    minimalStore = mockStore({});
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <CreateModelButton {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
  });

  test('should render with minimal props and store without exploding', () => {
    expect(wrapper.find(CreateModelButton).length).toBe(1);
  });

  test('should render button type link correctly', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <CreateModelButton buttonType={'link'} />
        </BrowserRouter>
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
