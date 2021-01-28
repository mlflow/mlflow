import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Form, Input } from 'antd';

export const NEW_NAME_FIELD = 'newName';

/**
 * Component that renders a form for updating a run's or experiment's name.
 */
class RenameFormComponent extends Component {
  static propTypes = {
    form: PropTypes.object.isRequired,
    type: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    visible: PropTypes.bool.isRequired,
    validator: PropTypes.func,
  };

  componentDidUpdate(prevProps) {
    this.autoFocus(prevProps);
    this.resetFields(prevProps);
  }

  autoFocusInputRef = (inputToAutoFocus) => {
    this.inputToAutoFocus = inputToAutoFocus;
    inputToAutoFocus.focus();
    inputToAutoFocus.select();
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
    if (prevProps.name !== this.props.name) {
      // reset input field to reset displayed initialValue
      this.props.form.resetFields([NEW_NAME_FIELD]);
    }
  };

  render() {
    const { getFieldDecorator } = this.props.form;
    return (
      <Form layout='vertical'>
        <Form.Item label={`New ${this.props.type} name`}>
          {getFieldDecorator(NEW_NAME_FIELD, {
            rules: [
              { required: true, message: `Please input a new name for the ${this.props.type}.` },
              { validator: this.props.validator },
            ],
            initialValue: this.props.name,
          })(
            <Input placeholder={`Input a ${this.props.type} name`} ref={this.autoFocusInputRef} />,
          )}
        </Form.Item>
      </Form>
    );
  }
}

export const RenameForm = Form.create()(RenameFormComponent);
