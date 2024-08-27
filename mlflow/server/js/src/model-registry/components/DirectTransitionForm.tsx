/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { Form } from 'antd';
import { Checkbox, LegacyTooltip } from '@databricks/design-system';
import React from 'react';
import { ACTIVE_STAGES, archiveExistingVersionToolTipText, Stages, StageTagComponents } from '../constants';
import { FormattedMessage, injectIntl } from 'react-intl';

type Props = {
  innerRef?: any;
  toStage?: string;
  intl: {
    formatMessage: (...args: any[]) => any;
  };
};

export class DirectTransitionFormImpl extends React.Component<Props> {
  render() {
    const { toStage, innerRef } = this.props;

    return (
      <Form ref={innerRef} className="model-version-update-form">
        {/* prettier-ignore */}
        {toStage && ACTIVE_STAGES.includes(toStage) && (
          <Form.Item name="archiveExistingVersions" initialValue="true" valuePropName="checked" preserve={false}>
            <Checkbox componentId="codegen_mlflow_app_src_model-registry_components_directtransitionform.tsx_56">
              <LegacyTooltip title={archiveExistingVersionToolTipText(toStage)}>
                <FormattedMessage
                  defaultMessage="Transition existing {currentStage} model versions to
                    {archivedStage}"
                  description="Description text for checkbox for archiving existing model versions
                    in the toStage for model version stage transition"
                  values={{
                    currentStage: StageTagComponents[toStage],
                    archivedStage: StageTagComponents[Stages.ARCHIVED],
                  }}
                />
                &nbsp;
              </LegacyTooltip>
            </Checkbox>
          </Form.Item>
        )}
      </Form>
    );
  }
}

// @ts-expect-error TS(2769): No overload matches this call.
export const DirectTransitionForm = injectIntl(DirectTransitionFormImpl);
