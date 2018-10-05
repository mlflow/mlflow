import React from 'react';
import { RequestStateWrapper } from './RequestStateWrapper';
import ErrorCodes from '../sdk/ErrorCodes';
import { ErrorWrapper } from '../Actions';
import { shallow } from 'enzyme';

const activeRequest = {
  id: 'a',
  active: true,
};

const completeRequest = {
  id: 'a',
  active: false,
  data: { run_id: "run_id" }
};

const errorRequest = {
  id: 'a',
  active: false,
  error: new ErrorWrapper({
    responseText: `{"error_code": "${ErrorCodes.RESOURCE_DOES_NOT_EXIST}"}`
  })
};

test("Renders loading page when requests are not complete", () => {
  const wrapper = shallow(
    <RequestStateWrapper
      requests={[activeRequest, completeRequest, errorRequest]}
    >
      <div>I am the child</div>
    </RequestStateWrapper>
  );
  expect(wrapper.find('.RequestStateWrapper-spinner')).toHaveLength(1);
});

test("Renders children when requests are complete", () => {
  const wrapper = shallow(
    <RequestStateWrapper
      requests={[completeRequest]}
    >
      <div className='child'>I am the child</div>
    </RequestStateWrapper>
  );
  expect(wrapper.find('div.child')).toHaveLength(1);
  expect(wrapper.find('div.child').text()).toContain("I am the child");
});

test("Throws exception if errorRenderFunc is not defined and wrapper has bad request.", () => {
  try {
    shallow(
      <RequestStateWrapper
        requests={[errorRequest]}
      >
        <div className='child'>I am the child</div>
      </RequestStateWrapper>
    );
  } catch (e) {
    expect(e.message).toContain("GOTO error boundary");
  }
});

test("Throws exception if errorRenderFunc returns undefined and wrapper has bad request.", () => {
  try {
    shallow(
      <RequestStateWrapper
        requests={[errorRequest]}
        errorRenderFunc={() => {
          return undefined;
        }}
      >
        <div className='child'>I am the child</div>
      </RequestStateWrapper>
    );
  } catch (e) {
    expect(e.message).toContain("GOTO error boundary");
  }
});

test("Renders errorRenderFunc if wrapper has bad request.", () => {
  const wrapper = shallow(
    <RequestStateWrapper
      requests={[errorRequest]}
      errorRenderFunc={(requests) => {
        expect(requests).toEqual([errorRequest]);
        return <div className='error'>Error!</div>;
      }}
    >
      <div className='child'>I am the child</div>
    </RequestStateWrapper>
  );
  expect(wrapper.find('div.error')).toHaveLength(1);
  expect(wrapper.find('div.error').text()).toContain("Error!");
});
