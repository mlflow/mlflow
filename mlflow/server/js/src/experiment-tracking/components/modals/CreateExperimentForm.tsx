/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';

import { injectIntl } from 'react-intl';
import { Input, LegacyForm } from '@databricks/design-system';

export const EXP_NAME_FIELD = 'experimentName';
export const ARTIFACT_LOCATION = 'artifactLocation';

type Props = {
  validator?: (...args: any[]) => any;
  intl: {
    formatMessage: (...args: any[]) => any;
  };
  innerRef: any;
};

/**
 * Component that renders a form for creating a new experiment.
 */
class CreateExperimentFormComponent extends Component<Props> {
  render() {
    return (
      // @ts-expect-error TS(2322): Type '{ children: Element[]; ref: any; layout: "ve... Remove this comment to see the full error message
      <LegacyForm ref={this.props.innerRef} layout="vertical">
        <LegacyForm.Item
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
            },
            {
              validator: this.props.validator,
            },
          ]}
        >
          <Input
            componentId="codegen_mlflow_app_src_experiment-tracking_components_modals_createexperimentform.tsx_51"
            placeholder={this.props.intl.formatMessage({
              defaultMessage: 'Input an experiment name',
              description: 'Input placeholder to enter experiment name for create experiment',
            })}
            autoFocus
          />
        </LegacyForm.Item>
        <LegacyForm.Item
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
            componentId="codegen_mlflow_app_src_experiment-tracking_components_modals_createexperimentform.tsx_71"
            placeholder={this.props.intl.formatMessage({
              defaultMessage: 'Input an artifact location (optional)',
              description: 'Input placeholder to enter artifact location for create experiment',
            })}
          />
        </LegacyForm.Item>
      </LegacyForm>
    );
  }
}

// @ts-expect-error TS(2769): No overload matches this call.
export const CreateExperimentForm = injectIntl(CreateExperimentFormComponent);
