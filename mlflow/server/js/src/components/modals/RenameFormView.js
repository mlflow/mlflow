import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Form, Input } from 'antd';

export const NEW_NAME_FIELD = 'newName';

/**
 * Component that renders a form for updating a run's or experiment's name.
 */
class RenameFormViewComponent extends Component {
  static propTypes = {
    form: PropTypes.object.isRequired,
    type: PropTypes.string.isRequired,
  }

  render() {
    // const validationSchema = getValidationSchema(this.props.type);
    const { getFieldDecorator } = this.props.form;
    return (
      <Form layout='vertical'>
        <Form.Item label={`New ${this.props.type} name`}>
          {getFieldDecorator(NEW_NAME_FIELD, {
            rules: [
              { required: true, message: `Please input a new name for the ${this.props.type}.`},
            ],
          })(<Input
              placeholder={`Input a ${this.props.type} name`}
              autoFocus
            />)
          }
        </Form.Item>
      </Form>
    );
  }
}

export const RenameFormView = Form.create()(RenameFormViewComponent);
