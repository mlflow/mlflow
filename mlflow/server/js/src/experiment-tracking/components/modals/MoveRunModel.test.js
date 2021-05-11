import React from 'react';
import { shallow } from 'enzyme';
import { MoveRunModalImpl } from './MoveRunModal';
import { NEW_ID_FIELD } from './ExperimentIdForm';

/**
 * Return a function that can be used to mock run move API requests, appending moved run IDs
 * to the provided list.
 * @param shouldFail: If true, the generated function will return a promise that always reject
 * @param moveddIdsList List to which to append IDs of moved runs
 * @returns {function(*=): Promise<any>}
 */
const getMockMoveRunApiFn = (shouldFail, movedIdsList) => {
  return (runId) => {
    return new Promise((resolve, reject) => {
      window.setTimeout(() => {
        if (shouldFail) {
          reject();
        } else {
          movedIdsList.push(runId);
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
      moveRunApi: getMockMoveRunApiFn(false, []),
    };
    wrapper = shallow(<MoveRunModalImpl {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
  });

  test('should move each selected run on submission', (done) => {
    const movedRunIds = [];
    const moveRunApi = getMockMoveRunApiFn(false, movedRunIds);
    const values = { NEW_ID_FIELD: '2' };
    wrapper = shallow(<MoveRunModalImpl {...{ ...minimalProps, moveRunApi }} />);
    instance = wrapper.instance();
    instance.handleSubmit(values).then(() => {
      expect(movedRunIds).toEqual(minimalProps.selectedRunIds);
      done();
    });
  });

  test('should show error modal if deletion fails', (done) => {
    const movedRunIds = [];
    const moveRunApi = getMockMoveRunApiFn(true, movedRunIds);
    const values = { NEW_ID_FIELD: '2' };
    wrapper = shallow(<MoveRunModalImpl {...{ ...minimalProps, moveRunApi }} />);
    instance = wrapper.instance();
    instance.handleSubmit(values).then(() => {
      expect(movedRunIds).toEqual([]);
      expect(minimalProps.openErrorModal).toBeCalled();
      done();
    });
  });
});
