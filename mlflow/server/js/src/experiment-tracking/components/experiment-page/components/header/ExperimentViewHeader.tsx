import React, { useCallback, useMemo } from 'react';
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
import { Link, useLocation, useNavigate } from '../../../../../common/utils/RoutingUtils';
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
import { useExperimentKind, isGenAIExperimentKind } from '../../../../utils/ExperimentKindUtils';
import { ExperimentViewManagementMenu } from './ExperimentViewManagementMenu';
import {
  shouldEnableExperimentPageSideTabs,
  shouldEnableWorkflowBasedNavigation,
} from '@mlflow/mlflow/src/common/utils/FeatureUtils';

import { ExperimentKind } from '../../../../constants';
import { useGetExperimentPageActiveTabByRoute } from '../../hooks/useGetExperimentPageActiveTabByRoute';
import { useWorkflowType } from '@mlflow/mlflow/src/common/contexts/WorkflowTypeContext';
import { getTabDisplayIcon, getTabDisplayName } from './ExperimentViewHeader.utils';

const getDocLinkHref = (experimentKind: ExperimentKind) => {
  if (isGenAIExperimentKind(experimentKind)) {
    return 'https://mlflow.org/docs/latest/genai/?rel=mlflow_ui';
  }
  return 'https://mlflow.org/docs/latest/ml/getting-started/?rel=mlflow_ui';
};

/**
 * Header for a single experiment page. Displays title, breadcrumbs and provides
 * controls for renaming, deleting and editing permissions.
 */
export const ExperimentViewHeader = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
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
    const navigate = useNavigate();
    const location = useLocation();
    const handleBack = useCallback(() => {
      const pathSegments = location.pathname.split('/').filter(Boolean);
      // Navigate to /experiments for tab pages (up to 3 segments: /experiments/ID/tab)
      // For deeper paths, remove last segment to navigate to parent
      if (pathSegments.length <= 3 && pathSegments[0] === 'experiments') {
        navigate(Routes.experimentsObservatoryRoute);
      } else {
        pathSegments.pop();
        navigate('/' + pathSegments.join('/'));
      }
    }, [location.pathname, navigate]);
    const experimentIds = useMemo(() => (experiment ? [experiment?.experimentId] : []), [experiment]);

    // In OSS, we don't need to show the docs link anymore as the link is in the sidebar
    const showDocsLink = false;

    // Extract the last part of the experiment name
    const { tabName: activeTabByRoute } = useGetExperimentPageActiveTabByRoute();
    const { workflowType } = useWorkflowType();
    const tabDisplayName = activeTabByRoute ? getTabDisplayName(activeTabByRoute, workflowType) : undefined;
    const normalizedExperimentName = useMemo(() => experiment.name.split('/').pop(), [experiment.name]);
    const experimentTitle =
      shouldEnableWorkflowBasedNavigation() && tabDisplayName ? tabDisplayName : normalizedExperimentName;

    const breadcrumbs: React.ReactNode[] = useMemo(
      () => [
        // eslint-disable-next-line react/jsx-key
        <Link to={Routes.experimentsObservatoryRoute} data-testid="experiment-observatory-link">
          <FormattedMessage
            defaultMessage="Experiments"
            description="Breadcrumb nav item to link to the list of experiments page"
          />
        </Link>,
        <Link to={Routes.getExperimentPageRoute(experiment.experimentId ?? '')} data-testid="experiment-link">
          {normalizedExperimentName}
        </Link>,
      ],
      [experiment.experimentId, normalizedExperimentName],
    );

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

    const experimentKindFromContext = useExperimentKind(experiment.tags);
    const experimentKind = inferredExperimentKind ?? experimentKindFromContext;
    const docLinkHref = getDocLinkHref(experimentKind ?? ExperimentKind.NO_INFERRED_TYPE);
    const showBreadcrumbs = !shouldEnableExperimentPageSideTabs() || shouldEnableWorkflowBasedNavigation();

    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          marginBottom: shouldEnableExperimentPageSideTabs() ? theme.spacing.xs : theme.spacing.sm,
        }}
      >
        {showBreadcrumbs && (
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
                {!shouldEnableWorkflowBasedNavigation() && (
                  <Button
                    componentId="mlflow.experiment-page.header.back-icon-button"
                    data-testid="experiment-view-header-back-button"
                    type="tertiary"
                    icon={<ArrowLeftIcon />}
                    onClick={handleBack}
                  />
                )}
                <div
                  css={{
                    borderRadius: theme.borders.borderRadiusSm,
                    backgroundColor: theme.colors.backgroundSecondary,
                    padding: theme.spacing.sm,
                  }}
                >
                  {getTabDisplayIcon(activeTabByRoute)}
                </div>
              </>
            )}
            <Tooltip
              content={normalizedExperimentName}
              open={shouldEnableWorkflowBasedNavigation() ? false : undefined}
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
                  {experimentTitle}
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
            {shouldEnableExperimentPageSideTabs() && (
              <ExperimentViewManagementMenu
                experiment={experiment}
                setEditing={setEditing}
                refetchExperiment={refetchExperiment}
              />
            )}
            <ExperimentViewHeaderShareButton
              type={shouldEnableExperimentPageSideTabs() ? undefined : 'primary'}
              experimentIds={experimentIds}
              searchFacetsState={searchFacetsState}
              uiState={uiState}
            />
            {shouldEnableExperimentPageSideTabs() && showDocsLink && (
              <Typography.Link
                componentId="mlflow.experiment-page.header.docs-link"
                href={docLinkHref}
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button componentId="mlflow.experiment-page.header.docs-link-button" icon={<InfoBookIcon />}>
                  <FormattedMessage
                    defaultMessage="View docs"
                    description="Text for docs link button on experiment view page header"
                  />
                </Button>
              </Typography.Link>
            )}
            {!shouldEnableExperimentPageSideTabs() && (
              <ExperimentViewManagementMenu
                experiment={experiment}
                setEditing={setEditing}
                refetchExperiment={refetchExperiment}
              />
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
