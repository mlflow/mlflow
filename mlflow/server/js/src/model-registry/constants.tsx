/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Tag } from '@databricks/design-system';
// @ts-expect-error TS(2306): File '/Users/elad.ossadon/universe4/mlflow/web/js/... Remove this comment to see the full error message
// eslint-disable-next-line
import * as overrides from './constant-overrides'; // eslint-disable-line import/no-namespace
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
  [Stages.NONE]: <Tag>{StageLabels[Stages.NONE]}</Tag>,
  [Stages.STAGING]: <Tag color="lemon">{StageLabels[Stages.STAGING]}</Tag>,
  [Stages.PRODUCTION]: <Tag color="lime">{StageLabels[Stages.PRODUCTION]}</Tag>,
  [Stages.ARCHIVED]: <Tag color="charcoal">{StageLabels[Stages.ARCHIVED]}</Tag>,
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
export const REGISTERED_MODELS_PER_PAGE = 10;

// Variant for compact tables (unified list pattern), this is
// going to become a default soon
export const REGISTERED_MODELS_PER_PAGE_COMPACT = 25;

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

export const archiveExistingVersionToolTipText = (currentStage: any) => (
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
