import React from 'react';
import { RequestStateWrapper, DEFAULT_ERROR_MESSAGE } from './RequestStateWrapper';
import { ErrorCodes } from '../constants';
import { shallow } from 'enzyme';
import { Spinner } from './Spinner';
import { ErrorWrapper } from '../utils/ActionUtils';

const activeRequest = {
  id: 'a',
  active: true,
};

const completeRequest = {
  id: 'a',
  active: false,
  data: { run_id: 'run_id' },
};

const errorRequest = {
  id: 'a',
  active: false,
  error: new ErrorWrapper({
    responseText: `{"error_code": "${ErrorCodes.RESOURCE_DOES_NOT_EXIST}"}`,
  }),
};

test('Renders loading page when requests are not complete', () => {
  const wrapper = shallow(
    <RequestStateWrapper requests={[activeRequest, completeRequest]}>
      <div>I am the child</div>
    </RequestStateWrapper>,
  );
  expect(wrapper.find(Spinner)).toHaveLength(1);
});

test('Renders custom loading page when requests are not complete', () => {
  const wrapper = shallow(
    <RequestStateWrapper
      requests={[activeRequest, completeRequest]}
      customSpinner={<h1 className='custom-spinner'>a custom spinner</h1>}
    >
      <div>I am the child</div>
    </RequestStateWrapper>,
  );
  expect(wrapper.find('h1.custom-spinner')).toHaveLength(1);
});

test('Renders children when requests are complete', () => {
  const wrapper = shallow(
    <RequestStateWrapper requests={[completeRequest]}>
      <div className='child'>I am the child</div>
    </RequestStateWrapper>,
  );
  expect(wrapper.find('div.child')).toHaveLength(1);
  expect(wrapper.find('div.child').text()).toContain('I am the child');
});

test('Throws exception if child is a React element and wrapper has bad request.', () => {
  try {
    shallow(
      <RequestStateWrapper requests={[errorRequest]}>
        <div className='child'>I am the child</div>
      </RequestStateWrapper>,
    );
  } catch (e) {
    expect(e.message).toContain(DEFAULT_ERROR_MESSAGE);
  }
});

test('Throws exception if errorRenderFunc returns undefined and wrapper has bad request.', () => {
  try {
    shallow(
      <RequestStateWrapper
        requests={[errorRequest]}
        errorRenderFunc={() => {
          return undefined;
        }}
      >
        <div className='child'>I am the child</div>
      </RequestStateWrapper>,
    );
    assert.fail();
  } catch (e) {
    expect(e.message).toContain(DEFAULT_ERROR_MESSAGE);
  }
});

test('Render func works if wrapper has bad request.', () => {
  const wrapper = shallow(
    <RequestStateWrapper requests={[activeRequest, completeRequest, errorRequest]}>
      {(isLoading, shouldRenderError, requests) => {
        if (shouldRenderError) {
          expect(requests).toEqual([activeRequest, completeRequest, errorRequest]);
          return <div className='error'>Error!</div>;
        }
        return <div className='child'>I am the child</div>;
      }}
    </RequestStateWrapper>,
  );
  expect(wrapper.find('div.error')).toHaveLength(1);
  expect(wrapper.find('div.error').text()).toContain('Error!');
});

test('Should call child if child is a function, even if no requests', () => {
  const childFunction = jest.fn();
  shallow(<RequestStateWrapper requests={[]}>{childFunction}</RequestStateWrapper>);

  expect(childFunction).toHaveBeenCalledTimes(1);
});
