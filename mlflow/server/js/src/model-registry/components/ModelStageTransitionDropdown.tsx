import React from 'react';
import { Dropdown, Menu, ChevronDownIcon, ArrowRightIcon } from '@databricks/design-system';
import {
  Stages,
  StageTagComponents,
  ActivityTypes,
  type PendingModelVersionActivity,
  ACTIVE_STAGES,
} from '../constants';
import { remove } from 'lodash';
import { FormattedMessage } from 'react-intl';
import type { ModelStageTransitionFormModalValues } from './ModelStageTransitionFormModal';
import { ModelStageTransitionFormModal } from './ModelStageTransitionFormModal';

type ModelStageTransitionDropdownProps = {
  currentStage?: string;
  permissionLevel?: string;
  onSelect?: (activity: PendingModelVersionActivity, comment?: string, archiveExistingVersions?: boolean) => void;
};

type ModelStageTransitionDropdownState = {
  confirmModalVisible: boolean;
  confirmingActivity: PendingModelVersionActivity | null;
  handleConfirm: ((values: ModelStageTransitionFormModalValues) => void) | undefined;
};

export class ModelStageTransitionDropdown extends React.Component<
  ModelStageTransitionDropdownProps,
  ModelStageTransitionDropdownState
> {
  static defaultProps = {
    currentStage: Stages.NONE,
  };

  state: ModelStageTransitionDropdownState = {
    confirmModalVisible: false,
    confirmingActivity: null,
    handleConfirm: undefined,
  };

  handleMenuItemClick = (activity: PendingModelVersionActivity) => {
    const { onSelect } = this.props;
    this.setState({
      confirmModalVisible: true,
      confirmingActivity: activity,
      handleConfirm:
        onSelect &&
        ((values: ModelStageTransitionFormModalValues) => {
          this.setState({ confirmModalVisible: false });

          if (values) {
            const { archiveExistingVersions = false } = values;
            // @ts-expect-error TS(2722): Cannot invoke an object which is possibly 'undefin... Remove this comment to see the full error message
            onSelect(activity, archiveExistingVersions);
            return;
          }
        }),
    });
  };

  handleConfirmModalCancel = () => {
    this.setState({ confirmModalVisible: false });
  };

  getNoneCurrentStages = (currentStage?: string) => {
    const stages = Object.values(Stages);
    remove(stages, (s) => s === currentStage);
    return stages;
  };

  getMenu() {
    const { currentStage } = this.props;
    const nonCurrentStages = this.getNoneCurrentStages(currentStage);
    return (
      <Menu>
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
              defaultMessage="Transition to"
              description="Text for transitioning a model version to a different stage under
                 dropdown menu in model version page"
            />
            &nbsp;&nbsp;&nbsp;
            <ArrowRightIcon />
            &nbsp;&nbsp;&nbsp;
            {StageTagComponents[stage]}
          </Menu.Item>
        ))}
      </Menu>
    );
  }

  renderConfirmModal() {
    const { confirmModalVisible, confirmingActivity, handleConfirm } = this.state;

    if (!confirmingActivity) {
      return null;
    }

    const allowArchivingExistingVersions =
      confirmingActivity.type === ActivityTypes.APPLIED_TRANSITION &&
      ACTIVE_STAGES.includes(confirmingActivity.to_stage);

    return (
      <ModelStageTransitionFormModal
        visible={confirmModalVisible}
        toStage={confirmingActivity.to_stage}
        onConfirm={handleConfirm}
        onCancel={this.handleConfirmModalCancel}
        transitionDescription={renderActivityDescription(confirmingActivity)}
        allowArchivingExistingVersions={allowArchivingExistingVersions}
      />
    );
  }

  render() {
    const { currentStage } = this.props;
    return (
      <span>
        <Dropdown overlay={this.getMenu()} trigger={['click']} className="mlflow-stage-transition-dropdown">
          <span>
            {StageTagComponents[currentStage ?? Stages.NONE]}
            <ChevronDownIcon css={{ cursor: 'pointer', marginLeft: -4 }} />
          </span>
        </Dropdown>
        {this.renderConfirmModal()}
      </span>
    );
  }
}

export const renderActivityDescription = (activity: PendingModelVersionActivity) => {
  if (activity) {
    return (
      <div>
        <FormattedMessage
          defaultMessage="Transition to"
          description="Text for activity description under confirmation modal for model
             version stage transition"
        />
        &nbsp;&nbsp;&nbsp;
        <ArrowRightIcon />
        &nbsp;&nbsp;&nbsp;
        {StageTagComponents[activity.to_stage]}
      </div>
    );
  }
  return null;
};
