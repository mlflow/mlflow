import React from 'react';
import { Tag } from 'antd';
import { ConstantOverrides } from './overrides/constant-overrides';

export const Stages = {
  NONE: 'None',
  STAGING: 'Staging',
  PRODUCTION: 'Production',
  ARCHIVED: 'Archived',
};

export const ACTIVE_STAGES = [
  Stages.STAGING,
  Stages.PRODUCTION,
];

export const StageLabels = {
  [Stages.NONE]: 'None',
  [Stages.STAGING]: 'Staging',
  [Stages.PRODUCTION]: 'Production',
  [Stages.ARCHIVED]: 'Archived',
};

export const StageTagComponents = {
  [Stages.NONE]: <Tag>{StageLabels[Stages.NONE]}</Tag>,
  [Stages.STAGING]: <Tag color='orange'>{StageLabels[Stages.STAGING]}</Tag>,
  [Stages.PRODUCTION]: <Tag color='green'>{StageLabels[Stages.PRODUCTION]}</Tag>,
  [Stages.ARCHIVED]: (
    <Tag color='#eee' style={{ color: '#333'}}>{StageLabels[Stages.ARCHIVED]}</Tag>
  ),
};

export const ActivityTypes = ConstantOverrides.ActivityTypes || {
  APPLIED_TRANSITION: 'APPLIED_TRANSITION',
};

// TODO(Zangr) Apply a more general override for all configs
export const IconByActivityType = ConstantOverrides.IconByActivityType || {};

export const EMPTY_CELL_PLACEHOLDER = <div style={{ marginTop: -12 }}>_</div>;

export const ModelVersionStatus = {
  READY: 'READY',
  PENDING_REGISTRATION: 'PENDING_REGISTRATION',
  FAILED_REGISTRATION: 'FAILED_REGISTRATION',
};

export const DefaultModelVersionStatusMessages = {
  [ModelVersionStatus.READY]: 'Ready.',
  [ModelVersionStatus.PENDING_REGISTRATION]: 'Registration pending...',
  [ModelVersionStatus.FAILED_REGISTRATION]: 'Registration failed.',
};

export const modelVersionStatusIconTooltips = {
  [ModelVersionStatus.READY]: 'Ready',
  [ModelVersionStatus.PENDING_REGISTRATION]: 'Registration pending',
  [ModelVersionStatus.FAILED_REGISTRATION]: 'Registration failed',
};

export const ModelVersionStatusIcons = {
  [ModelVersionStatus.READY]:
    <i className='far fa-check-circle icon-ready model-version-status-icon' />,
  [ModelVersionStatus.PENDING_REGISTRATION]:
    <i className='fa fa-spinner fa-spin icon-pending model-version-status-icon' />,
  [ModelVersionStatus.FAILED_REGISTRATION]:
    <i className='fa fa-exclamation-triangle icon-fail model-version-status-icon' />,
};

export const MODEL_VERSION_STATUS_POLL_INTERVAL = 5000;

export const REGISTER_DIALOG_DESCRIPTION = ConstantOverrides.REGISTER_DIALOG_DESCRIPTION ||
  'Once registered, the model will be available in the model registry and become public.';

export const MODEL_VERSION_DELETE_MENU_ITEM_DISABLED_TOOLTIP_TEXT = `Model versions in active 
stages cannot be deleted. To delete this model version, transition it to an inactive stage
(e.g., 'None' or 'Archived').`;

export const REGISTERED_MODEL_DELETE_MENU_ITEM_DISABLED_TOOLTIP_TEXT = `Registered models with
versions in active stages cannot be deleted. To delete this registered model, transition its
versions to inactive stages (e.g., 'None' or 'Archived').`;
