import React from 'react';
import { Dropdown, Menu, Icon, Modal } from 'antd';
import PropTypes from 'prop-types';
import { Stages, StageTagComponents, ActivityTypes } from '../constants';
import { DirectTransitionForm } from './DirectTransitionForm';
import _ from 'lodash';

export class ModelStageTransitionDropdown extends React.Component {
  static propTypes = {
    currentStage: PropTypes.string,
    onSelect: PropTypes.func,
  };

  static defaultProps = {
    currentStage: Stages.NONE,
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
    const stages = Object.values(Stages);
    _.remove(stages, (s) => s === currentStage);
    return stages;
  };

  getMenu() {
    const { currentStage, onSelect } = this.props;
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
            Transition to &nbsp;&nbsp;&nbsp;&nbsp;
            <i className='fas fa-long-arrow-alt-right' />
            &nbsp;&nbsp;&nbsp;&nbsp;
            {StageTagComponents[stage]}
          </Menu.Item>
        ))}
      </Menu>
    );
  }

  renderConfirmModal() {
    const { confirmModalVisible, confirmingActivity, handleConfirm } = this.state;
    if (confirmingActivity) {
      const formComponent = (
        <DirectTransitionForm ref={this.transitionFormRef} toStage={confirmingActivity.to_stage} />
      );
      return (
        <Modal
          title='Stage Transition'
          visible={confirmModalVisible}
          onOk={handleConfirm}
          onCancel={this.handleConfirmModalCancel}
        >
          {renderActivityDescription(confirmingActivity)}
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
            {StageTagComponents[currentStage]}
            <Icon type='down' style={{ cursor: 'pointer', marginLeft: -4 }} />
          </span>
        </Dropdown>
        {this.renderConfirmModal()}
      </span>
    );
  }
}

export const renderActivityDescription = (activity) => {
  if (activity) {
    return (
      <div>
        Transition to &nbsp;&nbsp;&nbsp;
        <i className='fas fa-long-arrow-alt-right' />
        &nbsp;&nbsp;&nbsp;&nbsp;
        {StageTagComponents[activity.to_stage]}
      </div>
    );
  }
  return null;
};
