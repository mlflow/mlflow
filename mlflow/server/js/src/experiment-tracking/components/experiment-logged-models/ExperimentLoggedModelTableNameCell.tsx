import { Overflow, ParagraphSkeleton, Tag, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import type { LoggedModelProto } from '../../types';
import { getStableColorForRun } from '../../utils/RunNameUtils';
import { RunColorPill } from '../experiment-page/components/RunColorPill';
import { ExperimentLoggedModelTableGroupCell } from './ExperimentLoggedModelTableGroupCell';
import type { LoggedModelsTableRow } from './ExperimentLoggedModelListPageTable.utils';
import {
  isLoggedModelDataGroupDataRow,
  isLoggedModelRow,
  LoggedModelsTableGroupingEnabledClass,
} from './ExperimentLoggedModelListPageTable.utils';
import { isSymbol } from 'lodash';
import { useExperimentLoggedModelRegisteredVersions } from './hooks/useExperimentLoggedModelRegisteredVersions';
import { FormattedMessage } from 'react-intl';
import React, { useMemo } from 'react';
import { shouldUnifyLoggedModelsAndRegisteredModels } from '@mlflow/mlflow/src/common/utils/FeatureUtils';

export const ExperimentLoggedModelTableNameCell = (props: { data: LoggedModelsTableRow }) => {
  const { theme } = useDesignSystemTheme();
  const { data } = props;
  const loggedModelData = isLoggedModelRow(data) ? (data as LoggedModelProto) : null;
  const loggedModels = useMemo(() => (loggedModelData ? [loggedModelData] : []), [loggedModelData]);

  const isUnifiedLoggedModelsEnabled = shouldUnifyLoggedModelsAndRegisteredModels();

  const { modelVersions: allModelVersions, isLoading } = useExperimentLoggedModelRegisteredVersions({
    loggedModels,
    checkAcl: Boolean(loggedModelData) && isUnifiedLoggedModelsEnabled,
  });

  if (isSymbol(data)) {
    return null;
  }

  if (isLoggedModelDataGroupDataRow(data)) {
    return <ExperimentLoggedModelTableGroupCell data={data} />;
  }

  // Filter to only show models that the user has access to
  const registeredModelVersions =
    isUnifiedLoggedModelsEnabled && allModelVersions ? allModelVersions.filter((model) => model.hasAccess) : [];

  const originalName = data.info?.name;

  // Build tooltip content for original logged model info
  const getTooltipContent = () => {
    if (!isUnifiedLoggedModelsEnabled || registeredModelVersions.length === 0) {
      return null;
    }

    const linkUrl =
      data.info?.experiment_id && data.info?.model_id
        ? Routes.getExperimentLoggedModelDetailsPageRoute(data.info.experiment_id, data.info.model_id)
        : null;

    if (!linkUrl) {
      return (
        <FormattedMessage
          defaultMessage="Original logged model: {originalName}"
          description="Tooltip text showing the original logged model name"
          values={{ originalName }}
        />
      );
    }

    return (
      <div>
        <FormattedMessage
          defaultMessage="Original logged model: {originalModelLink}"
          description="Tooltip text with link to the original logged model"
          values={{
            originalModelLink: (
              <Link to={linkUrl} css={{ color: 'inherit', textDecoration: 'underline' }}>
                {originalName}
              </Link>
            ),
          }}
        />
      </div>
    );
  };

  const tooltipContent = getTooltipContent();

  // Show loading spinner if ACL checking is in progress
  if (isLoading && isUnifiedLoggedModelsEnabled) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          width: '100%',
          [`.${LoggedModelsTableGroupingEnabledClass} &`]: {
            paddingLeft: theme.spacing.lg,
          },
        }}
      >
        <RunColorPill color={getStableColorForRun(data.info?.model_id || '')} />
        <ParagraphSkeleton label="Loading..." />
      </div>
    );
  }

  // If we have any registered models, show them; otherwise show original logged model
  if (registeredModelVersions.length > 0) {
    // If there's only one registered model, show it normally with the color pill
    if (registeredModelVersions.length === 1) {
      const primaryModel = registeredModelVersions[0];

      const content = (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            [`.${LoggedModelsTableGroupingEnabledClass} &`]: {
              paddingLeft: theme.spacing.lg,
            },
          }}
        >
          <RunColorPill color={getStableColorForRun(data.info?.model_id || '')} />
          <Link to={primaryModel.link} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            {primaryModel.displayedName}
            <Tag
              componentId="mlflow.logged_model.name_cell_version_tag"
              css={{ marginRight: 0, verticalAlign: 'middle' }}
            >
              v{primaryModel.version}
            </Tag>
          </Link>
        </div>
      );

      return tooltipContent ? (
        <Tooltip content={tooltipContent} componentId="mlflow.logged_model.name_cell_tooltip">
          {content}
        </Tooltip>
      ) : (
        content
      );
    }

    // If there are multiple registered models, show primary + overflow for the rest
    const content = (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          [`.${LoggedModelsTableGroupingEnabledClass} &`]: {
            paddingLeft: theme.spacing.lg,
          },
        }}
      >
        <RunColorPill color={getStableColorForRun(data.info?.model_id || '')} />
        <Overflow>
          {registeredModelVersions.map((modelVersion) => (
            <React.Fragment key={modelVersion.link}>
              <Link to={modelVersion.link} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                {modelVersion.displayedName}
                <Tag
                  componentId="mlflow.logged_model.name_cell_version_tag"
                  css={{ marginRight: 0, verticalAlign: 'middle' }}
                >
                  v{modelVersion.version}
                </Tag>
              </Link>
            </React.Fragment>
          ))}
        </Overflow>
      </div>
    );

    return tooltipContent ? (
      <Tooltip content={tooltipContent} componentId="mlflow.logged_model.name_cell_tooltip">
        {content}
      </Tooltip>
    ) : (
      content
    );
  }

  // Fallback to original logged model behavior
  const linkUrl =
    data.info?.experiment_id && data.info?.model_id
      ? Routes.getExperimentLoggedModelDetailsPageRoute(data.info.experiment_id, data.info.model_id)
      : null;

  if (!linkUrl) {
    return <>{originalName}</>;
  }

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        [`.${LoggedModelsTableGroupingEnabledClass} &`]: {
          paddingLeft: theme.spacing.lg,
        },
      }}
    >
      <RunColorPill color={getStableColorForRun(data.info?.model_id || '')} />
      <Link to={linkUrl}>{originalName}</Link>
    </div>
  );
};
