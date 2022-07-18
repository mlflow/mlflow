import React from 'react';
// eslint-disable-next-line
import { FormattedMessage } from 'react-intl';
import { ReadyIcon } from './utils';

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
  [ModelVersionStatus.READY]: (
    <FormattedMessage
      defaultMessage='Ready.'
      description='Default status message for model versions that are ready'
    />
  ),
};

export const modelVersionStatusIconTooltips = {
  [ModelVersionStatus.READY]: (
    <FormattedMessage
      defaultMessage='Ready'
      description='Tooltip text for ready model version status icon in model view page'
    />
  ),
};

export const ModelVersionStatusIcons = {
  [ModelVersionStatus.READY]: <ReadyIcon />,
};

export const MODEL_VERSION_STATUS_POLL_INTERVAL = 10000;

export const REGISTERED_MODELS_PER_PAGE = 10;

export const MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER = 75; // request size has a limit of 4KB

export const REGISTERED_MODELS_SEARCH_NAME_FIELD = 'name';

export const REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD = 'timestamp';

export const MODEL_SCHEMA_TENSOR_TYPE = 'tensor';

export const AntdTableSortOrder = {
  ASC: 'ascend',
  DESC: 'descend',
};

export const MODEL_VERSION_DELETE_MENU_ITEM_DISABLED_TOOLTIP_TEXT = `You cannot delete a model
version in an active stage. To delete this model version, transition it to the 'Archived' stage.`;

export const REGISTERED_MODEL_DELETE_MENU_ITEM_DISABLED_TOOLTIP_TEXT = `You cannot delete a
registered model with versions in active stages ('Staging' or 'Production'). To delete this
registered model, transition versions in active stages to the 'Archived' stage.`;

export const archiveExistingVersionToolTipText = (currentStage) => (
  <FormattedMessage
    defaultMessage='Model versions in the `{currentStage}` stage will be moved to the
       `Archived` stage.'
    description='Tooltip text for transitioning existing model versions in stage to archived
       in the model versions page'
    values={{ currentStage: currentStage }}
  />
);
