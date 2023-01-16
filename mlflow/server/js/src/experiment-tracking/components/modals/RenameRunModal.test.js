import React from 'react';
import { shallowWithInjectIntl } from '../../../common/utils/TestUtils';
import { RenameRunModalWithIntl } from './RenameRunModal';
import { GenericInputModal } from './GenericInputModal';

describe('RenameRunModal', () => {
  let wrapper;
  let minimalProps;
  let mockUpdateRunApi;

  beforeEach(() => {
    mockUpdateRunApi = jest.fn(() => Promise.resolve({}));
    minimalProps = {
      isOpen: false,
      runUuid: 'testUuid',
      runName: 'testName',
      onClose: jest.fn(() => Promise.resolve({})),
      updateRunApi: mockUpdateRunApi,
    };

    wrapper = shallowWithInjectIntl(<RenameRunModalWithIntl {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
    expect(wrapper.find(GenericInputModal).length).toBe(1);
  });

  test('test handleRenameRun closes modal in both success & failure cases', (done) => {
    const values = { newName: 'renamed' };
    const promise = wrapper.find(GenericInputModal).prop('handleSubmit')(values);
    promise.finally(() => {
      expect(mockUpdateRunApi).toHaveBeenCalledTimes(1);
      expect(mockUpdateRunApi).toHaveBeenCalledWith('testUuid', 'renamed', expect.any(String));
      done();
    });

    const mockFailUpdateRunApi = jest.fn(
      () =>
        new Promise((resolve, reject) => {
          window.setTimeout(() => {
            reject();
          }, 100);
        }),
    );
    const failProps = { ...minimalProps, updateRunApi: mockFailUpdateRunApi };
    const failWrapper = shallowWithInjectIntl(<RenameRunModalWithIntl {...failProps} />);
    const failPromise = failWrapper.find(GenericInputModal).prop('handleSubmit')(values);
    failPromise.finally(() => {
      expect(mockFailUpdateRunApi).toHaveBeenCalledTimes(1);
      expect(mockFailUpdateRunApi).toHaveBeenCalledWith('testUuid', 'renamed', expect.any(String));
      done();
    });
  });
});
