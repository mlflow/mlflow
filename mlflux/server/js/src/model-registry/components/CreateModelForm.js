import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Form, Input } from 'antd';
import { ModelRegistryDocUrl } from '../../common/constants';
import { FormattedMessage, injectIntl } from 'react-intl';

export const MODEL_NAME_FIELD = 'modelName';

/**
 * Component that renders a form for creating a new experiment.
 */
class CreateModelFormImpl extends Component {
  static propTypes = {
    form: PropTypes.object.isRequired,
    visible: PropTypes.bool.isRequired,
    validator: PropTypes.func,
    intl: PropTypes.any,
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
        <Form.Item
          label={this.props.intl.formatMessage({
            defaultMessage: 'Model name',
            description: 'Text for form title on creating model in the model registry',
          })}
        >
          {getFieldDecorator(MODEL_NAME_FIELD, {
            rules: [
              {
                required: true,
                message: this.props.intl.formatMessage({
                  defaultMessage: 'Please input a name for the new model.',
                  description:
                    'Error message for having no input for creating models in the model registry',
                }),
              },
              { validator: this.props.validator },
            ],
          })(<Input ref={this.autoFocusInputRef} />)}
        </Form.Item>
        <p className='create-modal-explanatory-text'>
          <FormattedMessage
            defaultMessage='After creation, you can register logged models as new versions.&nbsp;'
            description='Text for form description on creating model in the model registry'
          />
          <FormattedMessage
            defaultMessage='<link>Learn more</link>'
            description='Learn more link on the form for creating model in the model registry'
            values={{
              link: (chunks) => (
                <a href={learnMoreLinkUrl} target='_blank'>
                  {chunks}
                </a>
              ),
            }}
          />
          .
        </p>
      </Form>
    );
  }
}

export const CreateModelForm = Form.create()(injectIntl(CreateModelFormImpl));
