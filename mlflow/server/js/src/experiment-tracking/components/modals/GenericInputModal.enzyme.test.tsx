/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { shallow } from 'enzyme';
import { GenericInputModal } from './GenericInputModal';
import { Modal } from '@databricks/design-system';

class SimpleForm extends Component {
  render() {
    return null;
  }
}
function validateFields(isFieldValid: any) {
  if (!isFieldValid) {
    return Promise.reject(new Error("{ formField: 'formValue' }"));
  } else {
    return Promise.resolve({ formField: 'formValue' });
  }
}
function resetFields(resetFieldsFn: any) {
  resetFieldsFn();
}

describe('GenericInputModal', () => {
  let wrapper;
  let minimalProps: any;
  let resetFieldsMock: any;

  beforeEach(() => {
    resetFieldsMock = jest.fn();
    minimalProps = {
      isOpen: false,
      onClose: jest.fn(),
      onCancel: jest.fn(),
      // Mock submission handler that sleeps 1s then resolves
      handleSubmit: (values: any) =>
        new Promise((resolve) => {
          window.setTimeout(() => {
            // @ts-expect-error TS(2794): Expected 1 arguments, but got 0. Did you forget to... Remove this comment to see the full error message
            resolve();
          }, 1000);
        }),
      title: 'Enter your input',
      // @ts-expect-error TS(2769): No overload matches this call.
      children: <SimpleForm shouldValidationThrow={false} resetFieldsFn={resetFieldsMock} />,
    };
    wrapper = shallow(<GenericInputModal {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<GenericInputModal {...minimalProps} />);
    expect(wrapper.length).toBe(1);
    expect(wrapper.find(Modal).length).toBe(1);
  });

  test('should validate form contents and set submitting state in submission handler: successful submission case', async () => {
    // Test that validateFields() is called, and that handleSubmit is not called
    // when validation fails (and submitting state remains false)
    wrapper = shallow(<GenericInputModal {...minimalProps} />);
    const instance = wrapper.instance();
    wrapper.children(SimpleForm).props().innerRef.current = {
      validateFields: () => validateFields(true),
      resetFields: () => resetFields(resetFieldsMock),
    };
    const onValidationPromise = instance.onSubmit();
    expect(instance.state.isSubmitting).toEqual(true);
    await onValidationPromise;
    // We expect submission to succeed, and for the form fields to be reset and for the form to
    // no longer be submitting
    expect(resetFieldsMock).toHaveBeenCalled();
    expect(instance.state.isSubmitting).toEqual(false);
  });

  test('should validate form contents and set submitting state in submission handler: failed validation case', async () => {
    // Test that validateFields() is called, and that handleSubmit is not called
    // when validation fails (and submitting state remains false)
    // @ts-expect-error TS(2769): No overload matches this call.
    const form = <SimpleForm shouldValidationThrow resetFieldsFn={resetFieldsMock} />;
    const handleSubmit = jest.fn();
    wrapper = shallow(<GenericInputModal {...{ ...minimalProps, children: form, handleSubmit }} />);
    const instance = wrapper.instance();
    wrapper.children(SimpleForm).props().innerRef.current = {
      validateFields: () => validateFields(false),
      resetFields: () => resetFields(resetFieldsMock),
    };
    const onValidationPromise = instance.onSubmit();
    expect(instance.state.isSubmitting).toEqual(true);
    try {
      await onValidationPromise;
      // Reported during ESLint upgrade
      // eslint-disable-next-line no-undef, jest/no-jasmine-globals -- TODO: Fix this (use throw new Error())
      fail('Must throw');
    } catch (e) {
      // For validation errors, the form should not be reset (so that the user can fix the
      // validation error)
      expect(resetFieldsMock).not.toHaveBeenCalled();
      expect(handleSubmit).not.toHaveBeenCalled();
      expect(instance.state.isSubmitting).toEqual(false);
    }
  });

  // TODO: it seems that https://github.com/mlflow/mlflow/pull/15059 introduced test regression, to be investigated and fixed
  // eslint-disable-next-line jest/no-disabled-tests
  test.skip('should validate form contents and set submitting state in submission handler: failed submission case', async () => {
    // Test that validateFields() is called, and that handleSubmit is not called
    // when validation fails (and submitting state remains false)
    // @ts-expect-error TS(2769): No overload matches this call.
    const form = <SimpleForm shouldValidationThrow={false} resetFieldsFn={resetFieldsMock} />;
    const handleSubmit = (values: any) =>
      new Promise((resolve, reject) => {
        window.setTimeout(() => {
          reject(new Error());
        }, 1000);
      });
    wrapper = shallow(<GenericInputModal {...{ ...minimalProps, children: form, handleSubmit }} />);
    const instance = wrapper.instance();
    wrapper.children(SimpleForm).props().innerRef.current = {
      validateFields: () => validateFields(true),
      resetFields: () => resetFields(resetFieldsMock),
    };
    const onValidationPromise = instance.onSubmit();
    expect(instance.state.isSubmitting).toEqual(true);
    await onValidationPromise;
    // For validation errors, the form should not be reset (so that the user can fix the
    // validation error)
    expect(resetFieldsMock).toHaveBeenCalled();
    expect(instance.state.isSubmitting).toEqual(false);
  });
});
