/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { ConfirmModal } from './ConfirmModal';
import { Modal } from '@databricks/design-system';

describe('ConfirmModal', () => {
  let wrapper: any;
  let instance;
  let minimalProps: any;
  let onCloseMock: any;

  beforeEach(() => {
    onCloseMock = jest.fn();
    minimalProps = {
      isOpen: false,
      handleSubmit: jest.fn(() => Promise.resolve({})),
      onClose: onCloseMock,
      title: 'testTitle',
      helpText: 'testHelp',
      confirmButtonText: 'confirmTest',
    };
    wrapper = shallow(<ConfirmModal {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
    expect(wrapper.find(Modal).length).toBe(1);
  });

  test('onRequestCloseHandler executes properly based on state', () => {
    instance = wrapper.instance();
    instance.onRequestCloseHandler();
    expect(onCloseMock).toHaveBeenCalledTimes(1);

    instance.setState({ isSubmitting: true });
    instance.onRequestCloseHandler();
    expect(onCloseMock).toHaveBeenCalledTimes(1);
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('handleSubmitWrapper closes modal in both success & failure cases', (done) => {
    const promise = wrapper.find(Modal).prop('onOk')();
    promise.finally(() => {
      expect(onCloseMock).toHaveBeenCalledTimes(1);
      expect(wrapper.state('isSubmitting')).toBe(false);
      done();
    });

    const mockFailHandleSubmit = jest.fn(
      () =>
        new Promise((resolve, reject) => {
          window.setTimeout(() => {
            reject();
          }, 100);
        }),
    );
    const failProps = { ...minimalProps, handleSubmit: mockFailHandleSubmit };
    const failWrapper = shallow(<ConfirmModal {...failProps} />);
    const failPromise = failWrapper.find(Modal).prop('onOk')();
    failPromise.finally(() => {
      expect(onCloseMock).toHaveBeenCalledTimes(2);
      expect(failWrapper.state('isSubmitting')).toBe(false);
      done();
    });
  });
});
