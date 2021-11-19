import React, { Component } from 'react';
import PropTypes from 'prop-types';

import { injectIntl } from 'react-intl';
import { Input, Form } from 'antd';

export const EXP_NAME_FIELD = 'experimentName';
export const ARTIFACT_LOCATION = 'artifactLocation';

/**
 * Component that renders a form for creating a new experiment.
 */
class CreateExperimentFormComponent extends Component {
  static propTypes = {
    validator: PropTypes.func,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
    innerRef: PropTypes.any.isRequired,
  };

  render() {
    return (
      <Form ref={this.props.innerRef} layout='vertical'>
        <Form.Item
          label={this.props.intl.formatMessage({
            defaultMessage: 'Experiment Name',
            description: 'Label for create experiment modal to enter a valid experiment name',
          })}
          name={EXP_NAME_FIELD}
          rules={[
            {
              required: true,
              message: this.props.intl.formatMessage({
                defaultMessage: 'Please input a new name for the new experiment.',
                description: 'Error message for name requirement in create experiment for MLflow',
              }),
              validator: this.props.validator,
            },
          ]}
        >
          <Input
            placeholder={this.props.intl.formatMessage({
              defaultMessage: 'Input an experiment name',
              description: 'Input placeholder to enter experiment name for create experiment',
            })}
            autoFocus
          />
        </Form.Item>
        <Form.Item
          name={ARTIFACT_LOCATION}
          label={this.props.intl.formatMessage({
            defaultMessage: 'Artifact Location',
            description: 'Label for create experiment modal to enter a artifact location',
          })}
          rules={[
            {
              required: false,
            },
          ]}
        >
          <Input
            placeholder={this.props.intl.formatMessage({
              defaultMessage: 'Input an artifact location (optional)',
              description: 'Input placeholder to enter artifact location for create experiment',
            })}
          />
        </Form.Item>
      </Form>
    );
  }
}

export const CreateExperimentForm = injectIntl(CreateExperimentFormComponent);
