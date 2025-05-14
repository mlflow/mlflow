import React, { useMemo } from 'react';
import { Button, GenericSkeleton, NewWindowIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { PageHeader } from '../../../../../shared/building_blocks/PageHeader';
import { ExperimentViewCopyTitle } from './ExperimentViewCopyTitle';
import { ExperimentViewHeaderShareButton } from './ExperimentViewHeaderShareButton';
import { ExperimentEntity } from '../../../../types';
import { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { ExperimentViewArtifactLocation } from '../ExperimentViewArtifactLocation';
import { ExperimentViewCopyExperimentId } from './ExperimentViewCopyExperimentId';
import { ExperimentViewCopyArtifactLocation } from './ExperimentViewCopyArtifactLocation';
import { InfoIcon, InfoPopover } from '@databricks/design-system';
import { Popover } from '@databricks/design-system';
import { EXPERIMENT_PAGE_FEEDBACK_URL } from '@mlflow/mlflow/src/experiment-tracking/constants';

/**
 * Header for a single experiment page. Displays title, breadcrumbs and provides
 * controls for renaming, deleting and editing permissions.
 */
export const ExperimentViewHeader = React.memo(
  ({
    experiment,
    searchFacetsState,
    uiState,
    showAddDescriptionButton,
    setEditing,
  }: {
    experiment: ExperimentEntity;
    searchFacetsState?: ExperimentPageSearchFacetsState;
    uiState?: ExperimentPageUIState;
    showAddDescriptionButton: boolean;
    setEditing: (editing: boolean) => void;
  }) => {
    // eslint-disable-next-line prefer-const
    let breadcrumbs: React.ReactNode[] = [];
    const experimentIds = useMemo(() => (experiment ? [experiment?.experimentId] : []), [experiment]);

    const { theme } = useDesignSystemTheme();

    /**
     * Extract the last part of the experiment name
     */
    const normalizedExperimentName = useMemo(() => experiment.name.split('/').pop(), [experiment.name]);

    const feedbackFormUrl = EXPERIMENT_PAGE_FEEDBACK_URL;

    const renderFeedbackForm = () => {
      const feedbackLink = (
        <Button
          href={feedbackFormUrl}
          target="_blank"
          rel="noreferrer"
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_header_experimentviewheaderv2.tsx_100"
          css={{ marginLeft: theme.spacing.sm }}
          type="link"
          size="small"
          endIcon={<NewWindowIcon />}
        >
          <FormattedMessage
            defaultMessage="Provide Feedback"
            description="Link to a survey for users to give feedback"
          />
        </Button>
      );
      return feedbackLink;
    };

    const getShareButton = () => {
      const shareButtonElement = (
        <ExperimentViewHeaderShareButton
          experimentIds={experimentIds}
          searchFacetsState={searchFacetsState}
          uiState={uiState}
        />
      );
      return shareButtonElement;
    };

    const getInfoTooltip = () => {
      return (
        <div style={{ display: 'flex' }}>
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
    const getAddDescriptionButton = () => {
      return (
        <Button
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_header_experimentviewheaderv2.tsx_271"
          size="small"
          onClick={() => {
            setEditing(true);
          }}
          css={{
            marginLeft: theme.spacing.sm,
            background: `${theme.colors.backgroundSecondary} !important`,
            border: 'none',
          }}
        >
          <Typography.Text size="md">Add Description</Typography.Text>
        </Button>
      );
    };

    return (
      <PageHeader
        title={
          <div
            css={{
              [theme.responsive.mediaQueries.xs]: {
                display: 'inline',
                wordBreak: 'break-all',
              },
              [theme.responsive.mediaQueries.sm]: {
                display: 'inline-block',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                verticalAlign: 'middle',
              },
            }}
            title={normalizedExperimentName}
          >
            {normalizedExperimentName}
          </div>
        }
        titleAddOns={[
          getInfoTooltip(),
          renderFeedbackForm(),
          showAddDescriptionButton && getAddDescriptionButton(),
        ].filter(Boolean)}
        breadcrumbs={breadcrumbs}
        spacerSize="sm"
        dangerouslyAppendEmotionCSS={{
          [theme.responsive.mediaQueries.sm]: {
            // Do not wrap the title and buttons on >= small screens
            '& > div': {
              flexWrap: 'nowrap',
            },
            // The title itself should display elements horizontally
            h2: {
              display: 'flex',
              overflow: 'hidden',
            },
          },
        }}
      >
        <div css={{ display: 'flex', gap: theme.spacing.sm }}>
          {/* Wrap the buttons in a flex element */}
          {getShareButton()}
        </div>
      </PageHeader>
    );
  },
);

export function ExperimentViewHeaderSkeleton() {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ height: 2 * theme.general.heightSm }}>
      <div css={{ height: theme.spacing.lg }}>
        <GenericSkeleton css={{ width: 100, height: theme.spacing.md }} loading />
      </div>
      <div css={{ display: 'flex', justifyContent: 'space-between' }}>
        <div>
          <GenericSkeleton css={{ width: 160, height: theme.general.heightSm }} loading />
        </div>
        <div css={{ display: 'flex', gap: theme.spacing.sm }}>
          <GenericSkeleton css={{ width: 100, height: theme.general.heightSm }} loading />
          <GenericSkeleton css={{ width: 60, height: theme.general.heightSm }} loading />
        </div>
      </div>
    </div>
  );
}
