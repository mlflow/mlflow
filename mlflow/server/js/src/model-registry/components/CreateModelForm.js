import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Form, Input } from 'antd';
import { ModelRegistryDocUrl } from '../../common/constants';

export const MODEL_NAME_FIELD = 'modelName';

/**
 * Component that renders a form for creating a new experiment.
 */
class CreateModelFormImpl extends Component {
  static propTypes = {
    form: PropTypes.object.isRequired,
    visible: PropTypes.bool.isRequired,
    validator: PropTypes.func,
  };

  static getLearnMoreLinkUrl = () => ModelRegistryDocUrl;

  componentDidUpdate(prevProps) {
    this.autoFocus(prevProps);
  }

  autoFocusInputRef = (inputToAutoFocus) => {
    this.inputToAutoFocus = inputToAutoFocus;
    inputToAutoFocus.focus();
  };

  autoFocus = (prevProps) => {
    if (prevProps.visible === false && this.props.visible === true) {
      // focus on input field
      this.inputToAutoFocus && this.inputToAutoFocus.focus();
    }
  };

  render() {
    const { getFieldDecorator } = this.props.form;
    const learnMoreLinkUrl = CreateModelFormImpl.getLearnMoreLinkUrl();
    return (
      <Form layout='vertical'>
        <Form.Item label={'Model name'}>
          {getFieldDecorator(MODEL_NAME_FIELD, {
            rules: [
              { required: true, message: 'Please input a name for the new model.' },
              { validator: this.props.validator },
            ],
          })(<Input ref={this.autoFocusInputRef} />)}
        </Form.Item>
        <p className='create-modal-explanatory-text'>
          After creation, you can register logged models as new versions.&nbsp;
          <a target='_blank' href={learnMoreLinkUrl}>
            Learn more
          </a>
          .
        </p>
      </Form>
    );
  }
}

export const CreateModelForm = Form.create()(CreateModelFormImpl);
