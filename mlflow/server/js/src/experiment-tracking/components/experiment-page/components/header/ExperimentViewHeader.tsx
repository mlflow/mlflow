import React, { useMemo } from 'react';
import {
  ArrowLeftIcon,
  BeakerIcon,
  Breadcrumb,
  Button,
  InfoBookIcon,
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
import { getExperimentKindFromTags, isGenAIExperimentKind } from '../../../../utils/ExperimentKindUtils';
import { ExperimentViewManagementMenu } from './ExperimentViewManagementMenu';
import { shouldEnableExperimentPageSideTabs } from '@mlflow/mlflow/src/common/utils/FeatureUtils';

import { ExperimentKind } from '../../../../constants';

const GENAI_DOCS_PAGE_ROUTE = 'https://mlflow.org/docs/latest/genai/?rel=mlflow_ui';
const ML_DOCS_PAGE_ROUTE = 'https://mlflow.org/docs/latest/ml/getting-started/?rel=mlflow_ui';

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
    const docLinkHref = isGenAIExperimentKind(experimentKind ?? ExperimentKind.NO_INFERRED_TYPE)
      ? GENAI_DOCS_PAGE_ROUTE
      : ML_DOCS_PAGE_ROUTE;

    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          marginBottom: theme.spacing.sm,
        }}
      >
        {!shouldEnableExperimentPageSideTabs() && (
          <Breadcrumb includeTrailingCaret>
            {breadcrumbs.map((breadcrumb, index) => (
              <Breadcrumb.Item key={index}>{breadcrumb}</Breadcrumb.Item>
            ))}
          </Breadcrumb>
        )}
        <div
          css={{
            display: 'grid',
            gridTemplateColumns: shouldEnableExperimentPageSideTabs() ? '1fr auto auto' : '1fr 1fr 1fr',
          }}
        >
          <div
            css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center', overflow: 'hidden', minWidth: 250 }}
          >
            {shouldEnableExperimentPageSideTabs() && (
              <>
                <Link to={Routes.experimentsObservatoryRoute}>
                  <Button
                    componentId="mlflow.experiment-page.header.back-icon-button"
                    type="tertiary"
                    icon={<ArrowLeftIcon />}
                  />
                </Link>
                <div
                  css={{
                    borderRadius: theme.borders.borderRadiusSm,
                    backgroundColor: theme.colors.backgroundSecondary,
                    padding: theme.spacing.sm,
                  }}
                >
                  <BeakerIcon />
                </div>
              </>
            )}
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
          {shouldEnableExperimentPageSideTabs() ? <div /> : <TabSelectorBar experimentKind={experimentKind} />}
          <div
            css={{ display: 'flex', gap: theme.spacing.sm, justifyContent: 'flex-end', marginLeft: theme.spacing.sm }}
          >
            {!shouldEnableExperimentPageSideTabs() && (
              <ExperimentViewHeaderShareButton
                experimentIds={experimentIds}
                searchFacetsState={searchFacetsState}
                uiState={uiState}
                type="primary"
              />
            )}
            <ExperimentViewManagementMenu
              experiment={experiment}
              setEditing={setEditing}
              refetchExperiment={refetchExperiment}
            />
            {shouldEnableExperimentPageSideTabs() && (
              <>
                <ExperimentViewHeaderShareButton
                  experimentIds={experimentIds}
                  searchFacetsState={searchFacetsState}
                  uiState={uiState}
                />
                <Link to={docLinkHref} target="_blank" rel="noopener noreferrer">
                  <Button componentId="mlflow.experiment-page.header.docs-link-button" icon={<InfoBookIcon />}>
                    <FormattedMessage
                      defaultMessage="View docs"
                      description="Text for docs link button on experiment view page header"
                    />
                  </Button>
                </Link>
              </>
            )}
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
