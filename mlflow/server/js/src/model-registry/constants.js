import React from 'react';
import { Tag } from 'antd';
// eslint-disable-next-line
import * as overrides from './constant-overrides'; // eslint-disable-line import/no-namespace

export const Stages = {
  NONE: 'None',
  STAGING: 'Staging',
  PRODUCTION: 'Production',
  ARCHIVED: 'Archived',
};

export const ACTIVE_STAGES = [Stages.STAGING, Stages.PRODUCTION];

export const StageLabels = {
  [Stages.NONE]: 'None',
  [Stages.STAGING]: 'Staging',
  [Stages.PRODUCTION]: 'Production',
  [Stages.ARCHIVED]: 'Archived',
};

export const StageTagComponents = {
  [Stages.NONE]: <Tag key='none'>{StageLabels[Stages.NONE]}</Tag>,
  [Stages.STAGING]: (
    <Tag key='staging' className='staging-tag'>
      {StageLabels[Stages.STAGING]}
    </Tag>
  ),
  [Stages.PRODUCTION]: (
    <Tag key='production' className='production-tag'>
      {StageLabels[Stages.PRODUCTION]}
    </Tag>
  ),
  [Stages.ARCHIVED]: (
    <Tag key='archived' color='#eee' style={{ color: '#333' }}>
      {StageLabels[Stages.ARCHIVED]}
    </Tag>
  ),
};

export const ActivityTypes = {
  APPLIED_TRANSITION: 'APPLIED_TRANSITION',
  REQUESTED_TRANSITION: 'REQUESTED_TRANSITION',
  SYSTEM_TRANSITION: 'SYSTEM_TRANSITION',
  CANCELLED_REQUEST: 'CANCELLED_REQUEST',
  APPROVED_REQUEST: 'APPROVED_REQUEST',
  REJECTED_REQUEST: 'REJECTED_REQUEST',
  NEW_COMMENT: 'NEW_COMMENT',
};

export const EMPTY_CELL_PLACEHOLDER = <div style={{ marginTop: -12 }}>_</div>;

export const ModelVersionStatus = {
  READY: 'READY',
};

export const DefaultModelVersionStatusMessages = {
  [ModelVersionStatus.READY]: 'Ready.',
};

export const modelVersionStatusIconTooltips = {
  [ModelVersionStatus.READY]: 'Ready',
};

export const ModelVersionStatusIcons = {
  [ModelVersionStatus.READY]: (
    <i className='far fa-check-circle icon-ready model-version-status-icon' />
  ),
};

export const MODEL_VERSION_STATUS_POLL_INTERVAL = 10000;

export const REGISTERED_MODELS_PER_PAGE = 10;

export const MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER = 75; // request size has a limit of 4KB

export const REGISTERED_MODELS_SEARCH_NAME_FIELD = 'name';

export const REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD = 'timestamp';

export const AntdTableSortOrder = {
  ASC: 'ascend',
  DESC: 'descend',
};

export const MODEL_VERSION_DELETE_MENU_ITEM_DISABLED_TOOLTIP_TEXT = `You cannot delete a model
version in an active stage. To delete this model version, transition it to the 'Archived' stage.`;

export const REGISTERED_MODEL_DELETE_MENU_ITEM_DISABLED_TOOLTIP_TEXT = `You cannot delete a
registered model with versions in active stages ('Staging' or 'Production'). To delete this
registered model, transition versions in active stages to the 'Archived' stage.`;

export const archiveExistingVersionToolTipText = (currentStage) => `Model versions in the
'${currentStage}' stage will be moved to the 'Archived' stage.`;
