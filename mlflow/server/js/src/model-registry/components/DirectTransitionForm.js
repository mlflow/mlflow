import { Checkbox, Form, Tooltip } from 'antd';
import React from 'react';
import PropTypes from 'prop-types';
import {
  ACTIVE_STAGES,
  archiveExistingVersionToolTipText,
  Stages,
  StageTagComponents,
} from '../constants';
import { FormattedMessage, injectIntl } from 'react-intl';

export class DirectTransitionFormImpl extends React.Component {
  static propTypes = {
    form: PropTypes.object,
    toStage: PropTypes.string,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

  render() {
    const { toStage, form } = this.props;
    const { getFieldDecorator } = form;
    const archiveExistingVersionsCheckbox =
      toStage && ACTIVE_STAGES.includes(toStage) ? (
        <Form.Item>
          {getFieldDecorator('archiveExistingVersions', {
            initialValue: true,
            valuePropName: 'checked',
          })(
            <Checkbox>
              <Tooltip title={archiveExistingVersionToolTipText(toStage)}>
                <FormattedMessage
                  defaultMessage='Transition existing {currentStage} model versions to
                    {archivedStage}'
                  description='Description text for checkbox for archiving existing model versions
                    in the toStage for model version stage transition'
                  values={{
                    currentStage: StageTagComponents[toStage],
                    archivedStage: StageTagComponents[Stages.ARCHIVED],
                  }}
                />
                &nbsp;
              </Tooltip>
            </Checkbox>,
          )}
        </Form.Item>
      ) : (
        ''
      );

    return (
      <Form className='model-version-update-form'>
        {/* prettier-ignore */}
        {archiveExistingVersionsCheckbox}
      </Form>
    );
  }
}

export const DirectTransitionForm = Form.create()(injectIntl(DirectTransitionFormImpl));
