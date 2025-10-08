/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { DeleteRunModalImpl } from './DeleteRunModal';

/**
 * Return a function that can be used to mock run deletion API requests, appending deleted run IDs
 * to the provided list.
 * @param shouldFail: If true, the generated function will return a promise that always reject
 * @param deletedIdsList List to which to append IDs of deleted runs
 * @returns {function(*=): Promise<any>}
 */
const getMockDeleteRunApiFn = (shouldFail: any, deletedIdsList: any) => {
  return (runId: any) => {
    return new Promise((resolve, reject) => {
      window.setTimeout(() => {
        if (shouldFail) {
          reject();
        } else {
          deletedIdsList.push(runId);
          // @ts-expect-error TS(2794): Expected 1 arguments, but got 0. Did you forget to... Remove this comment to see the full error message
          resolve();
        }
      }, 1000);
    });
  };
};

describe('MyComponent', () => {
  let wrapper;
  let instance;
  let minimalProps: any;

  beforeEach(() => {
    minimalProps = {
      isOpen: false,
      onClose: jest.fn(),
      selectedRunIds: ['runId0', 'runId1'],
      openErrorModal: jest.fn(),
      deleteRunApi: getMockDeleteRunApiFn(false, []),
      intl: { formatMessage: jest.fn() },
      childRunIdsBySelectedParent: {},
    };
    wrapper = shallow(<DeleteRunModalImpl {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<DeleteRunModalImpl {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should delete each selected run on submission', (done) => {
    const deletedRunIds: any = [];
    const deleteRunApi = getMockDeleteRunApiFn(false, deletedRunIds);
    wrapper = shallow(<DeleteRunModalImpl {...{ ...minimalProps, deleteRunApi }} />);
    instance = wrapper.instance();
    instance.handleDeleteSelected().then(() => {
      expect(deletedRunIds).toEqual(minimalProps.selectedRunIds);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should show error modal if deletion fails', (done) => {
    const deletedRunIds: any = [];
    const deleteRunApi = getMockDeleteRunApiFn(true, deletedRunIds);
    wrapper = shallow(<DeleteRunModalImpl {...{ ...minimalProps, deleteRunApi }} />);
    instance = wrapper.instance();
    instance.handleDeleteSelected().then(() => {
      expect(deletedRunIds).toEqual([]);
      expect(minimalProps.openErrorModal).toHaveBeenCalled();
      done();
    });
  });

  test('should delete child runs when selected', (done) => {
    const deletedRunIds: any = [];
    const deleteRunApi = getMockDeleteRunApiFn(false, deletedRunIds);
    const childRunIdsBySelectedParent = { runId0: ['childRun1', 'childRun2'] };
    wrapper = shallow(<DeleteRunModalImpl {...{ ...minimalProps, deleteRunApi, childRunIdsBySelectedParent }} />);
    instance = wrapper.instance();
    instance.handleDeleteWithChildren().then(() => {
      expect(deletedRunIds).toEqual(['runId0', 'runId1', 'childRun1', 'childRun2']);
      done();
    });
  });

  test('should not duplicate already selected child runs', (done) => {
    const deletedRunIds: any = [];
    const deleteRunApi = getMockDeleteRunApiFn(false, deletedRunIds);
    const childRunIdsBySelectedParent = { runId0: ['runId1'] };
    wrapper = shallow(<DeleteRunModalImpl {...{ ...minimalProps, deleteRunApi, childRunIdsBySelectedParent }} />);
    instance = wrapper.instance();
    instance.handleDeleteWithChildren().then(() => {
      expect(deletedRunIds).toEqual(['runId0', 'runId1']);
      done();
    });
  });
});
