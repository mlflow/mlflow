import React from 'react';
import { DownOutlined } from '@ant-design/icons';
import { Dropdown, Menu, Modal } from 'antd';
import PropTypes from 'prop-types';
import  {ActivityTypes} from '../constants';

import { DirectTransitionForm } from './DirectTransitionForm';
import _ from 'lodash';
import { FormattedMessage } from 'react-intl';

export class ModelStageTransitionDropdown extends React.Component {
  static propTypes = {
    currentStage: PropTypes.string,
    onSelect: PropTypes.func,
    stageTagComponents: PropTypes.object,
    modelStageNames: PropTypes.array,
  };

  static defaultProps = {
    currentStage: "None", 
  };

  state = {
    confirmModalVisible: false,
    confirmingActivity: null,
    handleConfirm: undefined,
  };

  transitionFormRef = React.createRef();

  handleMenuItemClick = (activity) => {
    const { onSelect } = this.props;
    this.setState({
      confirmModalVisible: true,
      confirmingActivity: activity,
      handleConfirm:
        onSelect &&
        (() => {
          this.setState({ confirmModalVisible: false });
          const archiveExistingVersions = Boolean(
            this.transitionFormRef.current.getFieldValue('archiveExistingVersions'),
          );
          this.props.onSelect(activity, archiveExistingVersions);
        }),
    });
  };

  handleConfirmModalCancel = () => {
    this.setState({ confirmModalVisible: false });
  };

  getNoneCurrentStages = (currentStage) => {
    const stages = Object.keys(this.props.stageTagComponents);
    _.remove(stages, (s) => s === currentStage);
    return stages;
  };

  getMenu() {
    const { currentStage, onSelect, stageTagComponents } = this.props;
    const nonCurrentStages = this.getNoneCurrentStages(currentStage);

    return (
      <Menu onSelect={onSelect}>
        {nonCurrentStages.map((stage) => (
          <Menu.Item
            key={`transition-to-${stage}`}
            onClick={() =>
              this.handleMenuItemClick({
                type: ActivityTypes.APPLIED_TRANSITION,
                to_stage: stage,
              })
            }
          >
            <FormattedMessage
              defaultMessage='Transition to'
              description='Text for transitioning a model version to a different stage under
                 dropdown menu in model version page'
            />
            &nbsp;&nbsp;&nbsp;&nbsp;
            <i className='fas fa-long-arrow-alt-right' />
            &nbsp;&nbsp;&nbsp;&nbsp;
            {stageTagComponents[stage]}
          </Menu.Item>
        ))}
      </Menu>
    );
  }

  renderConfirmModal() {
    const { confirmModalVisible, confirmingActivity, handleConfirm } = this.state;
    if (confirmingActivity) {
      const formComponent = (
        <DirectTransitionForm
          innerRef={this.transitionFormRef}
          toStage={confirmingActivity.to_stage}
          availableStages={this.props.modelStageNames}
          stageTagComponents={this.props.stageTagComponents}
        />
      );
      return (
        <Modal
          title={
            <FormattedMessage
              defaultMessage='Stage Transition'
              description='Title text for model version stage transitions in confirm modal'
            />
          }
          visible={confirmModalVisible}
          onOk={handleConfirm}
          onCancel={this.handleConfirmModalCancel}
          okText={
            <FormattedMessage
              defaultMessage='OK'
              description='Text for OK button on the confirmation page for stage transition
                 on the model versions page'
            />
          }
          cancelText={
            <FormattedMessage
              defaultMessage='Cancel'
              description='Text for cancel button on the confirmation page for stage
                transitions on the model versions page'
            />
          }
        >
          {renderActivityDescription(confirmingActivity, this.props.stageTagComponents)}
          {formComponent}
        </Modal>
      );
    }
    return null;
  }

  render() {
    const { currentStage } = this.props;
    return (
      <span>
        <Dropdown
          overlay={this.getMenu()}
          trigger={['click']}
          className='stage-transition-dropdown'
        >
          <span>
            {this.props.stageTagComponents[currentStage]}
            <DownOutlined style={{ cursor: 'pointer', marginLeft: -4 }} />
          </span>
        </Dropdown>
        {this.renderConfirmModal()}
      </span>
    );
  }
}

export const renderActivityDescription = (activity, stageTagComponents) => {
  if (activity) {
    return (
      <div>
        <FormattedMessage
          defaultMessage='Transition to'
          description='Text for activity description under confirmation modal for model
             version stage transition'
        />
        &nbsp;&nbsp;&nbsp;
        <i className='fas fa-long-arrow-alt-right' />
        &nbsp;&nbsp;&nbsp;&nbsp;
        {stageTagComponents[activity.to_stage]}
      </div>
    );
  }
  return null;
};
