import React from 'react';
import { shallow } from 'enzyme';
import { RenameRunModalImpl } from './RenameRunModal';
import { GenericInputModal } from './GenericInputModal';
import Utils from '../../../common/utils/Utils';

describe('RenameRunModal', () => {
  let wrapper;
  let minimalProps;
  let mockSetTagApi;

  beforeEach(() => {
    mockSetTagApi = jest.fn(() => Promise.resolve({}));
    minimalProps = {
      isOpen: false,
      runUuid: 'testUuid',
      runName: 'testName',
      onClose: jest.fn(() => Promise.resolve({})),
      setTagApi: mockSetTagApi,
    };

    wrapper = shallow(<RenameRunModalImpl {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
    expect(wrapper.find(GenericInputModal).length).toBe(1);
  });

  test('test handleRenameRun closes modal in both success & failure cases', (done) => {
    const values = { newName: 'renamed' };
    const promise = wrapper.find(GenericInputModal).prop('handleSubmit')(values);
    promise.finally(() => {
      expect(mockSetTagApi).toHaveBeenCalledTimes(1);
      expect(mockSetTagApi).toHaveBeenCalledWith(
        'testUuid',
        Utils.runNameTag,
        'renamed',
        expect.any(String),
      );
      done();
    });

    const mockFailSetTagApi = jest.fn(
      () =>
        new Promise((resolve, reject) => {
          window.setTimeout(() => {
            reject();
          }, 100);
        }),
    );
    const failProps = { ...minimalProps, setTagApi: mockFailSetTagApi };
    const failWrapper = shallow(<RenameRunModalImpl {...failProps} />);
    const failPromise = failWrapper.find(GenericInputModal).prop('handleSubmit')(values);
    failPromise.finally(() => {
      expect(mockFailSetTagApi).toHaveBeenCalledTimes(1);
      expect(mockFailSetTagApi).toHaveBeenCalledWith(
        'testUuid',
        Utils.runNameTag,
        'renamed',
        expect.any(String),
      );
      done();
    });
  });
});
