import React, { useMemo } from 'react';
import {
  Breadcrumb,
  ParagraphSkeleton,
  TitleSkeleton,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../../../common/utils/RoutingUtils';
import Routes from '../../../../routes';
import { ExperimentViewCopyTitle } from './ExperimentViewCopyTitle';
import type { ExperimentEntity } from '../../../../types';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import type { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { ExperimentViewArtifactLocation } from '../ExperimentViewArtifactLocation';
import { ExperimentViewCopyExperimentId } from './ExperimentViewCopyExperimentId';
import { ExperimentViewCopyArtifactLocation } from './ExperimentViewCopyArtifactLocation';
import { InfoPopover } from '@databricks/design-system';
import { TabSelectorBar } from './tab-selector-bar/TabSelectorBar';
import { ExperimentViewHeaderShareButton } from './ExperimentViewHeaderShareButton';
import { getExperimentKindFromTags } from '../../../../utils/ExperimentKindUtils';
import { ExperimentViewManagementMenu } from './ExperimentViewManagementMenu';

import type { ExperimentKind } from '../../../../constants';
/**
 * Header for a single experiment page. Displays title, breadcrumbs and provides
 * controls for renaming, deleting and editing permissions.
 */
export const ExperimentViewHeader = React.memo(
  ({
    experiment,
    inferredExperimentKind,
    searchFacetsState,
    uiState,
    setEditing,
    experimentKindSelector,
    refetchExperiment,
  }: {
    experiment: ExperimentEntity;
    inferredExperimentKind?: ExperimentKind;
    searchFacetsState?: ExperimentPageSearchFacetsState;
    uiState?: ExperimentPageUIState;
    setEditing: (editing: boolean) => void;
    experimentKindSelector?: React.ReactNode;
    refetchExperiment?: () => Promise<unknown>;
  }) => {
    const { theme } = useDesignSystemTheme();
    const breadcrumbs: React.ReactNode[] = useMemo(
      () => [
        // eslint-disable-next-line react/jsx-key
        <Link to={Routes.experimentsObservatoryRoute} data-testid="experiment-observatory-link">
          <FormattedMessage
            defaultMessage="Experiments"
            description="Breadcrumb nav item to link to the list of experiments page"
          />
        </Link>,
      ],
      [],
    );
    const experimentIds = useMemo(() => (experiment ? [experiment?.experimentId] : []), [experiment]);
    // Extract the last part of the experiment name
    const normalizedExperimentName = useMemo(() => experiment.name.split('/').pop(), [experiment.name]);

    const getInfoTooltip = () => {
      return (
        <div style={{ display: 'flex', marginRight: theme.spacing.sm }}>
          <InfoPopover iconTitle="Info">
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.xs,
                flexWrap: 'nowrap',
              }}
              data-testid="experiment-view-header-info-tooltip-content"
            >
              <div style={{ whiteSpace: 'nowrap' }}>
                <FormattedMessage
                  defaultMessage="Path"
                  description="Label for displaying the current experiment path"
                />
                : {experiment.name + ' '}
                <ExperimentViewCopyTitle experiment={experiment} size="md" />
              </div>
              <div style={{ whiteSpace: 'nowrap' }}>
                <FormattedMessage
                  defaultMessage="Experiment ID"
                  description="Label for displaying the current experiment in view"
                />
                : {experiment.experimentId + ' '}
                <ExperimentViewCopyExperimentId experiment={experiment} />
              </div>
              <div style={{ whiteSpace: 'nowrap' }}>
                <FormattedMessage
                  defaultMessage="Artifact Location"
                  description="Label for displaying the experiment artifact location"
                />
                : <ExperimentViewArtifactLocation artifactLocation={experiment.artifactLocation} />{' '}
                <ExperimentViewCopyArtifactLocation experiment={experiment} />
              </div>
            </div>
          </InfoPopover>
        </div>
      );
    };

    const experimentKind = inferredExperimentKind ?? getExperimentKindFromTags(experiment.tags);

    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, marginBottom: theme.spacing.sm }}>
        <Breadcrumb includeTrailingCaret>
          {breadcrumbs.map((breadcrumb, index) => (
            <Breadcrumb.Item key={index}>{breadcrumb}</Breadcrumb.Item>
          ))}
        </Breadcrumb>
        <div css={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr' }}>
          <div
            css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center', overflow: 'hidden', minWidth: 250 }}
          >
            <Tooltip
              content={normalizedExperimentName}
              componentId="mlflow.experiment_view.header.experiment-name-tooltip"
            >
              <span
                css={{
                  maxWidth: '100%',
                  overflow: 'hidden',
                }}
              >
                <Typography.Title
                  withoutMargins
                  level={2}
                  css={{
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {normalizedExperimentName}
                </Typography.Title>
              </span>
            </Tooltip>
            {experimentKindSelector}
            {getInfoTooltip()}
          </div>
          <TabSelectorBar experimentKind={experimentKind} />
          <div
            css={{ display: 'flex', gap: theme.spacing.sm, justifyContent: 'flex-end', marginLeft: theme.spacing.sm }}
          >
            <ExperimentViewHeaderShareButton
              experimentIds={experimentIds}
              searchFacetsState={searchFacetsState}
              uiState={uiState}
            />
            <ExperimentViewManagementMenu
              experiment={experiment}
              setEditing={setEditing}
              refetchExperiment={refetchExperiment}
            />
          </div>
        </div>
      </div>
    );
  },
);

export function ExperimentViewHeaderSkeleton() {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <ParagraphSkeleton css={{ width: 100 }} loading />
      <div css={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr' }}>
        <TitleSkeleton css={{ width: 150, height: theme.general.heightSm }} loading />
        <TitleSkeleton css={{ height: theme.general.heightSm, alignSelf: 'center' }} loading />
        <TitleSkeleton css={{ width: theme.spacing.lg, height: theme.general.heightSm, alignSelf: 'right' }} loading />
      </div>
    </div>
  );
}
