/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { describe, beforeEach, jest, test, expect } from '@jest/globals';
import React from 'react';
import { shallow } from 'enzyme';
import { RestoreRunModalImpl } from './RestoreRunModal';
import { ConfirmModal } from './ConfirmModal';
import Utils from '../../../common/utils/Utils';

describe('RestoreRunModal', () => {
  let wrapper: any;
  let minimalProps: any;
  let mockRestoreRunApi: any;

  beforeEach(() => {
    mockRestoreRunApi = jest.fn(() => Promise.resolve({}));
    minimalProps = {
      isOpen: false,
      onClose: jest.fn(() => Promise.resolve({})),
      selectedRunIds: ['run1', 'run2'],
      restoreRunApi: mockRestoreRunApi,
    };

    wrapper = shallow(<RestoreRunModalImpl {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
    expect(wrapper.find(ConfirmModal).length).toBe(1);
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('handleRenameExperiment', (done) => {
    const promise = wrapper.find(ConfirmModal).prop('handleSubmit')();
    promise.finally(() => {
      expect(mockRestoreRunApi).toHaveBeenCalledTimes(2);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('handleRenameExperiment errors correctly', (done) => {
    const logErrorSpy = jest.spyOn(Utils, 'logErrorAndNotifyUser').mockImplementation(() => {});
    const mockFailRestoreRunApi = jest.fn(
      () =>
        new Promise((resolve, reject) => {
          window.setTimeout(() => {
            reject(new Error('Limit exceeded'));
          }, 1000);
        }),
    );
    const failRunApiProps = { ...minimalProps, restoreRunApi: mockFailRestoreRunApi };
    wrapper = shallow(<RestoreRunModalImpl {...failRunApiProps} />);

    const promise = wrapper.find(ConfirmModal).prop('handleSubmit')();
    promise.finally(() => {
      try {
        expect(mockFailRestoreRunApi).toHaveBeenCalledTimes(2);
        expect(logErrorSpy).toHaveBeenCalled();
      } finally {
        logErrorSpy.mockRestore();
        done();
      }
    });
  });
});
