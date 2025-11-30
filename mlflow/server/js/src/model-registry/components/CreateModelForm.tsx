/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';

import { LegacyForm, Input } from '@databricks/design-system';
import { ModelRegistryDocUrl } from '../../common/constants';
import { FormattedMessage, injectIntl } from 'react-intl';

export const MODEL_NAME_FIELD = 'modelName';

type Props = {
  visible: boolean;
  validator?: (...args: any[]) => any;
  intl?: any;
  innerRef: any;
};

/**
 * Component that renders a form for creating a new experiment.
 */
class CreateModelFormImpl extends Component<Props> {
  static getLearnMoreLinkUrl = () => ModelRegistryDocUrl;

  render() {
    const learnMoreLinkUrl = CreateModelFormImpl.getLearnMoreLinkUrl();
    return (
      // @ts-expect-error TS(2322)
      <LegacyForm ref={this.props.innerRef} layout="vertical" data-testid="create-model-form-modal">
        <LegacyForm.Item
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
                description: 'Error message for having no input for creating models in the model registry',
              }),
            },
            { validator: this.props.validator },
          ]}
        >
          <Input componentId="codegen_mlflow_app_src_model-registry_components_createmodelform.tsx_62" autoFocus />
        </LegacyForm.Item>
        <p className="create-modal-explanatory-text">
          <FormattedMessage
            defaultMessage="After creation, you can register logged models as new versions.&nbsp;"
            description="Text for form description on creating model in the model registry"
          />
          <FormattedMessage
            defaultMessage="<link>Learn more</link>"
            description="Learn more link on the form for creating model in the model registry"
            values={{
              link: (
                chunks: any, // Reported during ESLint upgrade
              ) => (
                // eslint-disable-next-line react/jsx-no-target-blank
                <a href={learnMoreLinkUrl} target="_blank">
                  {chunks}
                </a>
              ),
            }}
          />
          .
        </p>
      </LegacyForm>
    );
  }
}

// @ts-expect-error TS(2769): No overload matches this call.
export const CreateModelForm = injectIntl(CreateModelFormImpl);
