import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { shallow } from 'enzyme';
import { GenericInputModal } from './GenericInputModal';
import { Modal } from 'antd';

class SimpleForm extends Component {
  constructor(props) {
    super(props);
    this.resetFields = this.resetFields.bind(this);
    this.validateFields = this.validateFields.bind(this);
  }

  static propTypes = {
    shouldValidationThrow: PropTypes.bool.isRequired,
    resetFieldsFn: PropTypes.func.isRequired,
  };

  validateFields(errAndValuesFn) {
    if (this.props.shouldValidationThrow) {
      return errAndValuesFn('Form validation failed!', { formField: 'formValue' });
    } else {
      return errAndValuesFn(undefined, { formField: 'formValue' });
    }
  }

  resetFields() {
    this.props.resetFieldsFn();
  }

  render() {
    return null;
  }
}

describe('GenericInputModal', () => {
  let wrapper;
  let minimalProps;
  let resetFieldsMock;

  beforeEach(() => {
    resetFieldsMock = jest.fn();
    minimalProps = {
      isOpen: false,
      onClose: jest.fn(),
      // Mock submission handler that sleeps 1s then resolves
      handleSubmit: (values) =>
        new Promise((resolve) => {
          window.setTimeout(() => {
            resolve();
          }, 1000);
        }),
      title: 'Enter your input',
      children: <SimpleForm shouldValidationThrow={false} resetFieldsFn={resetFieldsMock} />,
    };
    wrapper = shallow(<GenericInputModal {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<GenericInputModal {...minimalProps} />);
    expect(wrapper.length).toBe(1);
    expect(wrapper.find(Modal).length).toBe(1);
  });

  test(
    'should validate form contents and set submitting state in submission handler: ' +
      'successful submission case',
    (done) => {
      // Test that validateFields() is called, and that handleSubmit is not called
      // when validation fails (and submitting state remains false)
      wrapper = shallow(<GenericInputModal {...minimalProps} />);
      const instance = wrapper.instance();
      // Call saveFormRef manually ourselves, since neither mount() nor shallow() seem to
      // result in saveFormRef being called
      instance.saveFormRef(shallow(minimalProps.children).instance());
      const promise = instance.onSubmit();
      expect(instance.state.isSubmitting).toEqual(true);
      promise.then(() => {
        // We expect submission to succeed, and for the form fields to be reset and for the form to
        // no longer be submitting
        expect(resetFieldsMock).toBeCalled();
        expect(instance.state.isSubmitting).toEqual(false);
        done();
      });
    },
  );

  test(
    'should validate form contents and set submitting state in submission handler: ' +
      'failed validation case',
    (done) => {
      // Test that validateFields() is called, and that handleSubmit is not called
      // when validation fails (and submitting state remains false)
      const form = <SimpleForm shouldValidationThrow resetFieldsFn={resetFieldsMock} />;
      const handleSubmit = jest.fn();
      wrapper = shallow(
        <GenericInputModal {...{ ...minimalProps, children: form, handleSubmit }} />,
      );
      const instance = wrapper.instance();
      // Call saveFormRef manually ourselves, since neither mount() nor shallow() seem to
      // result in saveFormRef being called
      instance.saveFormRef(shallow(form).instance());
      const promise = instance.onSubmit();
      expect(instance.state.isSubmitting).toEqual(false);
      promise.catch((e) => {
        // For validation errors, the form should not be reset (so that the user can fix the
        // validation error)
        expect(resetFieldsMock).not.toBeCalled();
        expect(handleSubmit).not.toBeCalled();
        expect(instance.state.isSubmitting).toEqual(false);
        done();
      });
    },
  );

  test(
    'should validate form contents and set submitting state in submission handler: ' +
      'failed submission case',
    (done) => {
      // Test that validateFields() is called, and that handleSubmit is not called
      // when validation fails (and submitting state remains false)
      const form = <SimpleForm shouldValidationThrow={false} resetFieldsFn={resetFieldsMock} />;
      const handleSubmit = (values) =>
        new Promise((resolve, reject) => {
          window.setTimeout(() => {
            reject();
          }, 1000);
        });
      wrapper = shallow(
        <GenericInputModal {...{ ...minimalProps, children: form, handleSubmit }} />,
      );
      const instance = wrapper.instance();
      // Call saveFormRef manually ourselves, since neither mount() nor shallow() seem to
      // result in saveFormRef being called
      instance.saveFormRef(shallow(form).instance());
      const promise = instance.onSubmit();
      expect(instance.state.isSubmitting).toEqual(true);
      promise.catch((e) => {
        // For validation errors, the form should not be reset (so that the user can fix the
        // validation error)
        expect(resetFieldsMock).toBeCalled();
        expect(instance.state.isSubmitting).toEqual(false);
        done();
      });
    },
  );
});
