import React from 'react';
import { shallow } from 'enzyme';
import { RenameExperimentModalImpl } from './RenameExperimentModal';
import { GenericInputModal } from './GenericInputModal';

describe('RenameExperimentModal', () => {
  let wrapper;
  let minimalProps;
  let mockUpdateExperimentApi;
  let mockGetExperimentApi;

  beforeEach(() => {
    mockUpdateExperimentApi = jest.fn(() => Promise.resolve({}));
    mockGetExperimentApi = jest.fn(() => Promise.resolve({}));
    minimalProps = {
      isOpen: false,
      experimentId: '123',
      experimentName: 'testName',
      experimentNames: ['arrayName1', 'arrayName2'],
      onClose: jest.fn(() => Promise.resolve({})),
      updateExperimentApi: mockUpdateExperimentApi,
      getExperimentApi: mockGetExperimentApi,
    };

    wrapper = shallow(<RenameExperimentModalImpl {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
    expect(wrapper.find(GenericInputModal).length).toBe(1);
  });

  test('form submission should result in updateExperimentApi and getExperimentApi calls', (done) => {
    const values = { newName: 'renamed' };
    const promise = wrapper.find(GenericInputModal).prop('handleSubmit')(values);
    promise.finally(() => {
      expect(mockUpdateExperimentApi).toHaveBeenCalledTimes(1);
      expect(mockUpdateExperimentApi).toHaveBeenCalledWith('123', 'renamed');
      expect(mockGetExperimentApi).toHaveBeenCalledTimes(1);
      done();
    });
  });

  test('if updateExperimentApi fails, getExperimentApi should not be called', (done) => {
    const values = { newName: 'renamed' };
    // Failing the updateExperimentApi call means getExperimentApi should not be called.
    const mockFailUpdateExperimentApi = jest.fn(
      () =>
        new Promise((resolve, reject) => {
          window.setTimeout(() => {
            reject();
          }, 100);
        }),
    );

    const failUpdateProps = { ...minimalProps, updateExperimentApi: mockFailUpdateExperimentApi };
    const failUpdateWrapper = shallow(<RenameExperimentModalImpl {...failUpdateProps} />);
    const failUpdatePromise = failUpdateWrapper.find(GenericInputModal).prop('handleSubmit')(
      values,
    );
    failUpdatePromise.catch(() => {
      expect(mockFailUpdateExperimentApi).toHaveBeenCalledTimes(1);
      expect(mockFailUpdateExperimentApi).toHaveBeenCalledWith('123', 'renamed');
      expect(mockGetExperimentApi).toHaveBeenCalledTimes(0);
      done();
    });
  });

  test('If getExperimentApi fails, updateExperimentApi should still have been called', (done) => {
    const values = { newName: 'renamed' };

    // Failing the getExperimentApi call should still mean both functions are called.
    const mockFailGetExperimentApi = jest.fn(
      () =>
        new Promise((resolve, reject) => {
          window.setTimeout(() => {
            reject();
          }, 100);
        }),
    );
    const failGetProps = { ...minimalProps, getExperimentApi: mockFailGetExperimentApi };
    const failGetWrapper = shallow(<RenameExperimentModalImpl {...failGetProps} />);
    const failGetPromise = failGetWrapper.find(GenericInputModal).prop('handleSubmit')(values);
    failGetPromise.catch(() => {
      expect(mockUpdateExperimentApi).toHaveBeenCalledTimes(1);
      expect(mockUpdateExperimentApi).toHaveBeenCalledWith('123', 'renamed');
      expect(mockFailGetExperimentApi).toHaveBeenCalledTimes(1);
      done();
    });
  });
});
