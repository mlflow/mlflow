import { Checkbox, Form, Input, Tooltip } from 'antd';
import React from 'react';
import PropTypes from 'prop-types';
import {
  ACTIVE_STAGES,
  archiveExistingVersionToolTipText,
  Stages,
  StageTagComponents,
} from '../constants';

const { TextArea } = Input;
export class DirectTransitionFormImpl extends React.Component {
  static propTypes = {
    form: PropTypes.object,
    toStage: PropTypes.string,
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
                Transition existing {StageTagComponents[toStage]}
                model versions to {StageTagComponents[Stages.ARCHIVED]}
              </Tooltip>
            </Checkbox>,
          )}
        </Form.Item>
      ) : '';

    return (
      <Form className='model-version-update-form'>
        <Form.Item label='Comment'>
          {getFieldDecorator('comment')(<TextArea rows={4} placeholder='Comment' />)}
        </Form.Item>
        {archiveExistingVersionsCheckbox}
      </Form>
    );
  }
}


export const DirectTransitionForm = Form.create()(DirectTransitionFormImpl);
