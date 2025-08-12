import React from 'react';
import { useState } from 'react';
import {
  Button,
  Drawer,
  Header,
  Spacer,
  TableIcon,
  Tag,
  LegacyTooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { RunDatasetWithTags } from '../../../../types';
import { MLFLOW_RUN_DATASET_CONTEXT_TAG } from '../../../../constants';
import { ExperimentViewDatasetSchema } from './ExperimentViewDatasetSchema';
import { ExperimentViewDatasetLink } from './ExperimentViewDatasetLink';
import { Link } from '../../../../../common/utils/RoutingUtils';
import Routes from '../../../../routes';
import { FormattedMessage } from 'react-intl';
import { ExperimentViewDatasetWithContext } from './ExperimentViewDatasetWithContext';
import { RunColorPill } from '../RunColorPill';
import { ExperimentViewDatasetSourceType } from './ExperimentViewDatasetSourceType';
import { ExperimentViewDatasetSourceURL } from './ExperimentViewDatasetSourceURL';
import { ExperimentViewDatasetDigest } from './ExperimentViewDatasetDigest';
import { useSelector } from 'react-redux';
import { ReduxState } from '../../../../../redux-types';
import { useGetExperimentRunColor } from '../../hooks/useExperimentRunColor';

export type DatasetWithRunType = {
  datasetWithTags: RunDatasetWithTags;
  runData: {
    experimentId?: string;
    tags?: Record<string, { key: string; value: string }>;
    runUuid: string;
    runName?: string;
    datasets: RunDatasetWithTags[];
  };
};

export interface DatasetsCellRendererProps {
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  selectedDatasetWithRun: DatasetWithRunType;
  setSelectedDatasetWithRun: (datasetWithRun: DatasetWithRunType) => void;
}

const DRAWER_WITDH = '800px';
const MAX_PROFILE_LENGTH = 80;

const areDatasetsEqual = (datasetA: RunDatasetWithTags, datasetB: RunDatasetWithTags) => {
  return datasetA.dataset.digest === datasetB.dataset.digest && datasetA.dataset.name === datasetB.dataset.name;
};

const ExperimentViewDatasetDrawerImpl = ({
  isOpen,
  setIsOpen,
  selectedDatasetWithRun,
  setSelectedDatasetWithRun,
}: DatasetsCellRendererProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const { datasetWithTags, runData } = selectedDatasetWithRun;
  const contextTag = selectedDatasetWithRun
    ? datasetWithTags?.tags?.find((tag) => tag.key === MLFLOW_RUN_DATASET_CONTEXT_TAG)
    : undefined;
  const fullProfile =
    datasetWithTags.dataset.profile && datasetWithTags.dataset.profile !== 'null'
      ? datasetWithTags.dataset.profile
      : undefined;

  const getRunColor = useGetExperimentRunColor();
  const { experimentId = '', tags = {} } = runData;

  return (
    <Drawer.Root
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) {
          setIsOpen(false);
        }
      }}
    >
      <Drawer.Content
        componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetdrawer.tsx_81"
        title={
          <div css={{ display: 'flex', alignItems: 'center', height: '100%' }}>
            <Typography.Title level={4} css={{ marginRight: theme.spacing.sm, marginBottom: 0 }}>
              <FormattedMessage
                defaultMessage="Data details for "
                description="Text for data details for the experiment run in the dataset drawer"
              />
            </Typography.Title>
            <Link to={Routes.getRunPageRoute(experimentId, runData.runUuid)} css={styles.runLink}>
              <RunColorPill color={getRunColor(runData.runUuid)} />
              <span css={styles.runName}>{runData.runName}</span>
            </Link>
          </div>
        }
        width={DRAWER_WITDH}
        footer={<Spacer size="xs" />}
      >
        <div
          css={{
            display: 'flex',
            borderTop: `1px solid ${theme.colors.border}`,
            height: '100%',
            marginLeft: -theme.spacing.sm,
          }}
        >
          {/* column for dataset selection */}
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              width: '300px',
              borderRight: `1px solid ${theme.colors.border}`,
              height: '100%',
            }}
          >
            <Typography.Text
              color="secondary"
              css={{
                marginBottom: theme.spacing.sm,
                marginTop: theme.spacing.sm,
                paddingLeft: theme.spacing.sm,
              }}
            >
              {runData.datasets.length}{' '}
              <FormattedMessage
                defaultMessage="datasets used"
                description="Text for dataset count in the experiment run dataset drawer"
              />
            </Typography.Text>
            <div
              css={{
                height: '100%',
                display: 'flex',
                overflow: 'auto',
              }}
              onWheel={(e) => e.stopPropagation()}
            >
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'visible',
                  flex: 1,
                }}
              >
                {runData.datasets.map((dataset) => (
                  <Typography.Link
                    componentId="mlflow.dataset_drawer.dataset_link"
                    aria-label={`${dataset.dataset.name} (${dataset.dataset.digest})`}
                    key={`${dataset.dataset.name}-${dataset.dataset.digest}`}
                    css={{
                      display: 'flex',
                      whiteSpace: 'nowrap',
                      textDecoration: 'none',
                      cursor: 'pointer',
                      flexDirection: 'column',
                      justifyContent: 'center',
                      alignItems: 'flex-start',
                      backgroundColor: areDatasetsEqual(dataset, datasetWithTags)
                        ? theme.colors.actionTertiaryBackgroundPress
                        : 'transparent',
                      paddingBottom: theme.spacing.sm,
                      paddingTop: theme.spacing.sm,
                      paddingLeft: theme.spacing.sm,
                      border: 0,
                      borderTop: `1px solid ${theme.colors.border}`,
                      '&:hover': {
                        backgroundColor: theme.colors.actionTertiaryBackgroundHover,
                      },
                    }}
                    onClick={() => {
                      setSelectedDatasetWithRun({ datasetWithTags: dataset, runData: runData });
                      setIsOpen(true);
                    }}
                  >
                    <ExperimentViewDatasetWithContext datasetWithTags={dataset} displayTextAsLink={false} />
                  </Typography.Link>
                ))}
              </div>
            </div>
          </div>
          {/* column for dataset details */}
          <div
            css={{
              overflow: 'hidden',
              paddingLeft: theme.spacing.md,
              paddingTop: theme.spacing.md,
              display: 'flex',
              flexDirection: 'column',
              width: '100%',
            }}
          >
            {/* dataset metadata */}
            <div
              css={{
                display: 'flex',
                gap: theme.spacing.sm,
              }}
            >
              <div css={{ flex: '1' }}>
                <Header
                  title={
                    <div css={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
                      <TableIcon css={{ marginRight: theme.spacing.xs }} />
                      <LegacyTooltip title={datasetWithTags.dataset.name}>
                        <Typography.Title ellipsis level={3} css={{ marginBottom: 0, maxWidth: 200 }}>
                          {datasetWithTags.dataset.name}
                        </Typography.Title>
                      </LegacyTooltip>
                      {contextTag && (
                        <Tag
                          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetdrawer.tsx_206"
                          css={{
                            textTransform: 'capitalize',
                            marginLeft: theme.spacing.xs,
                            marginRight: theme.spacing.xs,
                          }}
                        >
                          {contextTag.value}
                        </Tag>
                      )}
                    </div>
                  }
                />
                <Typography.Title
                  level={4}
                  color="secondary"
                  css={{ marginBottom: theme.spacing.xs, marginTop: theme.spacing.xs }}
                  title={fullProfile}
                >
                  {datasetWithTags.dataset.profile && datasetWithTags.dataset.profile !== 'null' ? (
                    datasetWithTags.dataset.profile.length > MAX_PROFILE_LENGTH ? (
                      `${datasetWithTags.dataset.profile.substring(0, MAX_PROFILE_LENGTH)} ...`
                    ) : (
                      datasetWithTags.dataset.profile
                    )
                  ) : (
                    <FormattedMessage
                      defaultMessage="No profile available"
                      description="Text for no profile available in the experiment run dataset drawer"
                    />
                  )}
                </Typography.Title>
              </div>
              <ExperimentViewDatasetLink datasetWithTags={datasetWithTags} runTags={tags} />
            </div>
            <div css={{ flexShrink: 0, display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <ExperimentViewDatasetDigest datasetWithTags={datasetWithTags} />
              <ExperimentViewDatasetSourceType datasetWithTags={datasetWithTags} />
              <ExperimentViewDatasetSourceURL datasetWithTags={datasetWithTags} />
            </div>
            {/* dataset schema */}
            <div
              css={{
                marginTop: theme.spacing.sm,
                marginBottom: theme.spacing.xs,
                borderTop: `1px solid ${theme.colors.border}`,
                opacity: 0.5,
              }}
            />
            <ExperimentViewDatasetSchema datasetWithTags={datasetWithTags} />
          </div>
        </div>
      </Drawer.Content>
    </Drawer.Root>
  );
};

// Memoize the component so it rerenders only when props change directly, preventing
// rerenders caused e.g. by the overarching context provider.
export const ExperimentViewDatasetDrawer = React.memo(ExperimentViewDatasetDrawerImpl);

const styles = {
  runLink: {
    overflow: 'hidden',
    display: 'flex',
    gap: 8,
    alignItems: 'center',
  },
  runName: {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    fontSize: '13px',
  },
};
