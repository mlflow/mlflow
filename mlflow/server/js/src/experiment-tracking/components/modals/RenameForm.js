import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Input, Form } from 'antd';

export const NEW_NAME_FIELD = 'newName';

/**
 * Component that renders a form for updating a run's or experiment's name.
 */
class RenameFormComponent extends Component {
  static propTypes = {
    type: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    visible: PropTypes.bool.isRequired,
    validator: PropTypes.func,
    innerRef: PropTypes.any.isRequired,
  };

  componentDidUpdate(prevProps) {
    this.autoFocus(prevProps);
    this.resetFields(prevProps);
  }

  autoFocusInputRef = (inputToAutoFocus) => {
    this.inputToAutoFocus = inputToAutoFocus;
    inputToAutoFocus && inputToAutoFocus.focus();
    inputToAutoFocus && inputToAutoFocus.select();
  };

  autoFocus = (prevProps) => {
    if (prevProps.visible === false && this.props.visible === true) {
      // focus on input field
      this.inputToAutoFocus && this.inputToAutoFocus.focus();
      // select text
      this.inputToAutoFocus && this.inputToAutoFocus.select();
    }
  };

  resetFields = (prevProps) => {
    const formRef = this.props.innerRef;
    if (prevProps.name !== this.props.name) {
      // reset input field to reset displayed initialValue
      formRef.current.resetFields([NEW_NAME_FIELD]);
    }
  };

  render() {
    return (
      <Form ref={this.props.innerRef} layout='vertical'>
        <Form.Item
          name={NEW_NAME_FIELD}
          initialValue={this.props.name}
          rules={[
            { required: true, message: `Please input a new name for the ${this.props.type}.` },
            { validator: this.props.validator },
          ]}
          label={`New ${this.props.type} name`}
        >
          <Input placeholder={`Input a ${this.props.type} name`} ref={this.autoFocusInputRef} />
        </Form.Item>
      </Form>
    );
  }
}

export const RenameForm = RenameFormComponent;
