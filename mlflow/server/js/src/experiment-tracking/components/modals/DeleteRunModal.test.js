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
const getMockDeleteRunApiFn = (shouldFail, deletedIdsList) => {
  return (runId) => {
    return new Promise((resolve, reject) => {
      window.setTimeout(() => {
        if (shouldFail) {
          reject();
        } else {
          deletedIdsList.push(runId);
          resolve();
        }
      }, 1000);
    });
  };
};

describe('MyComponent', () => {
  let wrapper;
  let instance;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      isOpen: false,
      onClose: jest.fn(),
      selectedRunIds: ['runId0', 'runId1'],
      openErrorModal: jest.fn(),
      deleteRunApi: getMockDeleteRunApiFn(false, []),
    };
    wrapper = shallow(<DeleteRunModalImpl {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<DeleteRunModalImpl {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should delete each selected run on submission', (done) => {
    const deletedRunIds = [];
    const deleteRunApi = getMockDeleteRunApiFn(false, deletedRunIds);
    wrapper = shallow(<DeleteRunModalImpl {...{ ...minimalProps, deleteRunApi }} />);
    instance = wrapper.instance();
    instance.handleSubmit().then(() => {
      expect(deletedRunIds).toEqual(minimalProps.selectedRunIds);
      done();
    });
  });

  test('should show error modal if deletion fails', (done) => {
    const deletedRunIds = [];
    const deleteRunApi = getMockDeleteRunApiFn(true, deletedRunIds);
    wrapper = shallow(<DeleteRunModalImpl {...{ ...minimalProps, deleteRunApi }} />);
    instance = wrapper.instance();
    instance.handleSubmit().then(() => {
      expect(deletedRunIds).toEqual([]);
      expect(minimalProps.openErrorModal).toBeCalled();
      done();
    });
  });
});
