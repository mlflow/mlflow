import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { Input, Form } from 'antd';
import { ModelRegistryDocUrl } from '../../common/constants';
import { FormattedMessage, injectIntl } from 'react-intl';

export const MODEL_NAME_FIELD = 'modelName';

/**
 * Component that renders a form for creating a new experiment.
 */
class CreateModelFormImpl extends Component {
  static propTypes = {
    visible: PropTypes.bool.isRequired,
    validator: PropTypes.func,
    intl: PropTypes.any,
    innerRef: PropTypes.any.isRequired,
  };

  static getLearnMoreLinkUrl = () => ModelRegistryDocUrl;

  render() {
    const learnMoreLinkUrl = CreateModelFormImpl.getLearnMoreLinkUrl();
    return (
      <Form ref={this.props.innerRef} layout='vertical'>
        <Form.Item
          name={MODEL_NAME_FIELD}
          label={this.props.intl.formatMessage({
            defaultMessage: 'Model name',
            description: 'Text for form title on creating model in the model registry',
          })}
          rules={[
            {
              required: true,
              message: this.props.intl.formatMessage({
                defaultMessage: 'Please input a name for the new model.',
                description:
                  'Error message for having no input for creating models in the model registry',
              }),
            },
            { validator: this.props.validator },
          ]}
        >
          <Input autoFocus />
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
                // Reported during ESLint upgrade
                // eslint-disable-next-line react/jsx-no-target-blank
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

export const CreateModelForm = injectIntl(CreateModelFormImpl);
