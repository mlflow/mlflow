/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { RequestStateWrapper, DEFAULT_ERROR_MESSAGE } from './RequestStateWrapper';
import { ErrorCodes } from '../constants';
import { shallow } from 'enzyme';
import { Spinner } from './Spinner';
import { ErrorWrapper } from '../utils/ErrorWrapper';

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
  id: 'errorId',
  active: false,
  error: new ErrorWrapper(`{"error_code": "${ErrorCodes.RESOURCE_DOES_NOT_EXIST}"}`, 404),
};

const non404ErrorRequest = {
  id: 'errorId2',
  active: false,
  error: new ErrorWrapper(`{"error_code": "${ErrorCodes.INTERNAL_ERROR}"}`, 500),
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
      customSpinner={<h1 className="custom-spinner">a custom spinner</h1>}
    >
      <div>I am the child</div>
    </RequestStateWrapper>,
  );
  expect(wrapper.find('h1.custom-spinner')).toHaveLength(1);
});

test('Renders children when requests are complete', () => {
  const wrapper = shallow(
    <RequestStateWrapper requests={[completeRequest]}>
      <div className="child">I am the child</div>
    </RequestStateWrapper>,
  );
  expect(wrapper.find('div.child')).toHaveLength(1);
  expect(wrapper.find('div.child').text()).toContain('I am the child');
});

test('Throws exception if child is a React element and wrapper has bad request.', () => {
  try {
    shallow(
      <RequestStateWrapper requests={[errorRequest]}>
        <div className="child">I am the child</div>
      </RequestStateWrapper>,
    );
  } catch (e) {
    expect((e as any).message).toContain(DEFAULT_ERROR_MESSAGE);
  }
});

test('Throws exception if errorRenderFunc returns undefined and wrapper has bad request.', () => {
  try {
    shallow(
      <RequestStateWrapper requests={[errorRequest]}>
        <div className="child">I am the child</div>
      </RequestStateWrapper>,
    );
    // @ts-expect-error TS(2304): Cannot find name 'assert'.
    assert.fail();
  } catch (e) {
    expect((e as any).message).toContain(DEFAULT_ERROR_MESSAGE);
  }
});

test('Renders child if request expectedly returns a 404', () => {
  const wrapper = shallow(
    <RequestStateWrapper requests={[errorRequest]} requestIdsWith404sToIgnore={[errorRequest.id]}>
      <div className="child">I am the child</div>
    </RequestStateWrapper>,
  );
  expect(wrapper.find('div.child')).toHaveLength(1);
  expect(wrapper.find('div.child').text()).toContain('I am the child');
});

test('Does not render child if request returns a non-404 error', () => {
  try {
    shallow(
      <RequestStateWrapper requests={[non404ErrorRequest]} requestIdsWith404sToIgnore={[errorRequest.id]}>
        <div className="child">I am the child</div>
      </RequestStateWrapper>,
    );
    // @ts-expect-error TS(2304): Cannot find name 'assert'.
    assert.fail();
  } catch (e) {
    expect((e as any).message).toContain(DEFAULT_ERROR_MESSAGE);
  }
});

test('Render func works if wrapper has bad request.', () => {
  const wrapper = shallow(
    <RequestStateWrapper requests={[activeRequest, completeRequest, errorRequest]}>
      {(isLoading: any, shouldRenderError: any, requests: any) => {
        if (shouldRenderError) {
          expect(requests).toEqual([activeRequest, completeRequest, errorRequest]);
          return <div className="error">Error!</div>;
        }
        return <div className="child">I am the child</div>;
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
