import React from 'react';
import { shallowWithInjectIntl } from '../../../common/utils/TestUtils';
import { MoveRunsModalWithIntl } from './MoveRunsModal';
import { GenericInputModal } from './GenericInputModal';

describe('MoveRunsModal', () => {
  let wrapper;
  let minimalProps;
  let mockMoveRunsApi;

  beforeEach(() => {
    mockMoveRunsApi = jest.fn(() => Promise.resolve({}));
    minimalProps = {
      isOpen: false,
      experimentList: [{ experiment_id: '1' }, { experiment_id: '2' }],
      selectedRunIds: [1, 2],
      onClose: jest.fn(() => Promise.resolve({})),
      moveRunsApi: mockMoveRunsApi,
    };

    wrapper = shallowWithInjectIntl(<MoveRunsModalWithIntl {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
    expect(wrapper.find(GenericInputModal).length).toBe(1);
  });

  test('test handleMoveRuns closes modal in both success & failure cases', (done) => {
    const values = { runIds: [1, 2], experimentId: '1' };
    const promise = wrapper.find(GenericInputModal).prop('handleSubmit')(values);
    promise.finally(() => {
      expect(mockMoveRunsApi).toHaveBeenCalledTimes(1);
      expect(mockMoveRunsApi).toHaveBeenCalledWith([1, 2], '1');
      done();
    });

    const mockFailMoveRunsApi = jest.fn(
      () =>
        new Promise((resolve, reject) => {
          window.setTimeout(() => {
            reject();
          }, 100);
        }),
    );
    const failProps = { ...minimalProps, moveRunsApi: mockFailMoveRunsApi };
    const failWrapper = shallowWithInjectIntl(<MoveRunsModalWithIntl {...failProps} />);
    const failPromise = failWrapper.find(GenericInputModal).prop('handleSubmit')(values);
    failPromise.finally(() => {
      expect(mockFailMoveRunsApi).toHaveBeenCalledTimes(1);
      expect(mockFailMoveRunsApi).toHaveBeenCalledWith([1, 2], '1', expect.any(String));
      done();
    });
  });
});
