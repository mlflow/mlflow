import React, { useMemo } from 'react';
import { Theme } from '@emotion/react';
import { Button, NewWindowIcon, InfoTooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { PageHeader } from '../../../../../shared/building_blocks/PageHeader';
import { ExperimentViewCopyTitle } from './ExperimentViewCopyTitle';
import { ExperimentViewHeaderShareButton } from './ExperimentViewHeaderShareButton';
import { ExperimentEntity } from '../../../../types';
import { useExperimentPageFeedbackUrl } from '../../hooks/useExperimentPageFeedbackUrl';
import { ExperimentPageSearchFacetsStateV2 } from '../../models/ExperimentPageSearchFacetsStateV2';
import { ExperimentPageUIStateV2 } from '../../models/ExperimentPageUIStateV2';
import { ExperimentViewArtifactLocation } from '../ExperimentViewArtifactLocation';
import { ExperimentViewCopyExperimentId } from './ExperimentViewCopyExperimentId';
import { ExperimentViewCopyArtifactLocation } from './ExperimentViewCopyArtifactLocation';
import { Tooltip } from '@databricks/design-system';
import { InfoIcon } from '@databricks/design-system';
import { Popover } from '@databricks/design-system';

/**
 * Header for a single experiment page. Displays title, breadcrumbs and provides
 * controls for renaming, deleting and editing permissions.
 */
export const ExperimentViewHeaderV2 = React.memo(
  ({
    experiment,
    searchFacetsState,
    uiState,
    showAddDescriptionButton,
    setEditing,
  }: {
    experiment: ExperimentEntity;
    searchFacetsState?: ExperimentPageSearchFacetsStateV2;
    uiState?: ExperimentPageUIStateV2;
    showAddDescriptionButton: boolean;
    setEditing: (editing: boolean) => void;
  }) => {
    // eslint-disable-next-line prefer-const
    let breadcrumbs: React.ReactNode[] = [];
    const experimentIds = useMemo(() => (experiment ? [experiment?.experiment_id] : []), [experiment]);

    const { theme } = useDesignSystemTheme();

    /**
     * Extract the last part of the experiment name
     */
    const normalizedExperimentName = useMemo(() => experiment.name.split('/').pop(), [experiment.name]);

    const feedbackFormUrl = useExperimentPageFeedbackUrl();

    const renderFeedbackForm = () => {
      const feedbackLink = (
        <a href={feedbackFormUrl} target="_blank" rel="noreferrer" css={{ display: 'flex', alignItems: 'center' }}>
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_header_experimentviewheaderv2.tsx_100"
            css={{ marginLeft: theme.spacing.sm }}
            type="link"
            size="small"
          >
            <FormattedMessage
              defaultMessage="Provide Feedback"
              description="Link to a survey for users to give feedback"
            />
          </Button>
          <NewWindowIcon css={{ marginLeft: theme.spacing.xs }} />
        </a>
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
          <Tooltip
            placement="bottomLeft"
            dangerouslySetAntdProps={{ overlayStyle: { maxWidth: 'none' } }}
            arrowPointAtCenter
            title={
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
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
                  : {experiment.experiment_id + ' '}
                  <ExperimentViewCopyExperimentId experiment={experiment} />
                </div>
                <div style={{ whiteSpace: 'nowrap' }}>
                  <FormattedMessage
                    defaultMessage="Artifact Location"
                    description="Label for displaying the experiment artifact location"
                  />
                  : <ExperimentViewArtifactLocation artifactLocation={experiment.artifact_location} />{' '}
                  <ExperimentViewCopyArtifactLocation experiment={experiment} />
                </div>
              </div>
            }
          >
            <InfoIcon css={{ color: theme.colors.textSecondary }} />
          </Tooltip>
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

    const HEADER_MAX_WIDTH = '70%';

    return (
      <PageHeader
        title={
          <div
            css={{
              display: 'inline-block',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              maxWidth: HEADER_MAX_WIDTH,
              textOverflow: 'ellipsis',
              verticalAlign: 'middle',
            }}
          >
            {normalizedExperimentName}
          </div>
        }
        /* prettier-ignore */
        titleAddOns={[
          getInfoTooltip(),
          feedbackFormUrl && renderFeedbackForm(),
          showAddDescriptionButton && getAddDescriptionButton(),
        ]}
        breadcrumbs={breadcrumbs}
        spacerSize="sm"
      >
        {getShareButton()}
      </PageHeader>
    );
  },
);
