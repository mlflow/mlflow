import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Form, Input } from 'antd';

export const EXP_NAME_FIELD = 'experimentName';

/**
 * Component that renders a form for creating a new experiment.
 */
class CreateExperimentFormComponent extends Component {
  static propTypes = {
    form: PropTypes.object.isRequired,
    visible: PropTypes.bool.isRequired,
  }

  componentDidUpdate(prevProps) {
    this.autoFocus(prevProps);
  }

  autoFocusInputRef = (inputToAutoFocus) => {
    this.inputToAutoFocus = inputToAutoFocus;
    inputToAutoFocus.focus();
  }

  autoFocus = (prevProps) => {
    if (prevProps.visible === false && this.props.visible === true) {
      // focus on input field
      this.inputToAutoFocus && this.inputToAutoFocus.focus();
    }
  }

  render() {
    // const validationSchema = getValidationSchema(this.props.type);
    const { getFieldDecorator } = this.props.form;
    return (
      <Form layout='vertical'>
        <Form.Item label={`Experiment Name`}>
          {getFieldDecorator(EXP_NAME_FIELD, {
            rules: [
              { required: true, message: `Please input a new name for the new experiment.`},
            ],
          })(<Input
              placeholder='Input an experiment name'
              ref={this.autoFocusInputRef}
            />)
          }
        </Form.Item>
      </Form>
    );
  }
}

export const CreateExperimentForm = Form.create()(CreateExperimentFormComponent);
