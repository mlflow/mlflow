import { Tag } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ReadyIcon } from './utils';

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
  [Stages.NONE]: (
    <Tag componentId="codegen_mlflow_app_src_model-registry_constants.tsx_37">{StageLabels[Stages.NONE]}</Tag>
  ),
  [Stages.STAGING]: (
    <Tag componentId="codegen_mlflow_app_src_model-registry_constants.tsx_38" color="lemon">
      {StageLabels[Stages.STAGING]}
    </Tag>
  ),
  [Stages.PRODUCTION]: (
    <Tag componentId="codegen_mlflow_app_src_model-registry_constants.tsx_39" color="lime">
      {StageLabels[Stages.PRODUCTION]}
    </Tag>
  ),
  [Stages.ARCHIVED]: (
    <Tag componentId="codegen_mlflow_app_src_model-registry_constants.tsx_40" color="charcoal">
      {StageLabels[Stages.ARCHIVED]}
    </Tag>
  ),
};

export interface ModelVersionActivity {
  creation_timestamp?: number;
  user_id?: string;
  activity_type: ActivityTypes;
  comment?: string;
  last_updated_timestamp?: number;
  from_stage?: string;
  to_stage?: string;
  system_comment?: string;
  id?: string;
}

export enum ActivityTypes {
  APPLIED_TRANSITION = 'APPLIED_TRANSITION',
  REQUESTED_TRANSITION = 'REQUESTED_TRANSITION',
  SYSTEM_TRANSITION = 'SYSTEM_TRANSITION',
  CANCELLED_REQUEST = 'CANCELLED_REQUEST',
  APPROVED_REQUEST = 'APPROVED_REQUEST',
  REJECTED_REQUEST = 'REJECTED_REQUEST',
  NEW_COMMENT = 'NEW_COMMENT',
}

export interface PendingModelVersionActivity {
  type: ActivityTypes;
  to_stage: string;
}

export const EMPTY_CELL_PLACEHOLDER = <div style={{ marginTop: -12 }}>_</div>;

export const ModelVersionStatus = {
  READY: 'READY',
};

export const DefaultModelVersionStatusMessages = {
  [ModelVersionStatus.READY]: (
    <FormattedMessage defaultMessage="Ready." description="Default status message for model versions that are ready" />
  ),
};

export const modelVersionStatusIconTooltips = {
  [ModelVersionStatus.READY]: (
    <FormattedMessage
      defaultMessage="Ready"
      description="Tooltip text for ready model version status icon in model view page"
    />
  ),
};

export const ModelVersionStatusIcons = {
  [ModelVersionStatus.READY]: <ReadyIcon />,
};

export const MODEL_VERSION_STATUS_POLL_INTERVAL = 10000;

// Number of registered models initially shown on the model registry list page
const REGISTERED_MODELS_PER_PAGE = 10;

// Variant for compact tables (unified list pattern), this is
// going to become a default soon
export const REGISTERED_MODELS_PER_PAGE_COMPACT = 25;
export const MODEL_VERSIONS_PER_PAGE_COMPACT = 25;

export const MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER = 75; // request size has a limit of 4KB

export const REGISTERED_MODELS_SEARCH_NAME_FIELD = 'name';

export const REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD = 'timestamp';

export const MODEL_VERSIONS_SEARCH_TIMESTAMP_FIELD = 'creation_timestamp';

export const AntdTableSortOrder = {
  ASC: 'ascend',
  DESC: 'descend',
};

export const archiveExistingVersionToolTipText = (currentStage: string) => (
  <FormattedMessage
    defaultMessage="Model versions in the `{currentStage}` stage will be moved to the
     `Archived` stage."
    description="Tooltip text for transitioning existing model versions in stage to archived
     in the model versions page"
    values={{ currentStage: currentStage }}
  />
);

export const mlflowAliasesLearnMoreLink =
  'https://mlflow.org/docs/latest/model-registry.html#using-registered-model-aliases';
