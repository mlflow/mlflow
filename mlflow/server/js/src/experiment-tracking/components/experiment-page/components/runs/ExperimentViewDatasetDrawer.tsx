import React from 'react';
import { useState } from 'react';
import {
  Button,
  Drawer,
  Header,
  Spacer,
  TableIcon,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { RunDatasetWithTags } from '../../../../types';
import { MLFLOW_RUN_DATASET_CONTEXT_TAG } from '../../../../constants';
import { Divider } from 'antd';
import { ExperimentViewDatasetSchema } from './ExperimentViewDatasetSchema';
import { ExperimentViewDatasetLink } from './ExperimentViewDatasetLink';
import { Link } from '../../../../../common/utils/RoutingUtils';
import Routes from '../../../../routes';
import { FormattedMessage } from 'react-intl';
import { ExperimentViewDatasetWithContext } from './ExperimentViewDatasetWithContext';

export type DatasetWithRunType = {
  datasetWithTags: RunDatasetWithTags;
  runData: {
    experimentId: string;
    tags: Record<string, { key: string; value: string }>;
    runUuid: string;
    runName: string;
    color?: string;
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

export const ExperimentViewDatasetDrawerImpl = ({
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
        title={
          <div css={{ display: 'flex', alignItems: 'center', height: '100%' }}>
            <Typography.Title level={4} css={{ marginRight: theme.spacing.sm, marginBottom: 0 }}>
              <FormattedMessage
                defaultMessage='Data details for '
                description='Text for data details for the experiment run in the dataset drawer'
              />
            </Typography.Title>
            <Link
              to={Routes.getRunPageRoute(runData.experimentId, runData.runUuid)}
              css={styles.runLink}
            >
              <div
                data-testid='dataset-drawer-run-color'
                css={{ ...styles.colorPill, backgroundColor: runData.color }}
              />
              <span css={styles.runName}>{runData.runName}</span>
            </Link>
          </div>
        }
        width={DRAWER_WITDH}
        footer={<Spacer size='xs' />}
      >
        <div
          css={{
            display: 'flex',
            borderTop: `1px solid ${theme.colors.border}`,
            height: '100%',
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
              color='secondary'
              css={{
                marginBottom: theme.spacing.sm,
                marginTop: theme.spacing.sm,
              }}
            >
              {runData.datasets.length}{' '}
              <FormattedMessage
                defaultMessage='datasets used'
                description='Text for dataset count in the experiment run dataset drawer'
              />
            </Typography.Text>
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                height: '100%',
                overflow: 'auto',
              }}
              onWheel={(e) => e.stopPropagation()}
            >
              {runData.datasets.map((dataset) => (
                <div
                  key={`${dataset.dataset.name}-${dataset.dataset.digest}`}
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    backgroundColor:
                      dataset.dataset.name === datasetWithTags.dataset.name &&
                      dataset.dataset.digest === datasetWithTags.dataset.digest
                        ? theme.colors.grey100
                        : 'transparent',
                    borderTop: `1px solid ${theme.colors.border}`,
                    borderBottom: `1px solid ${theme.colors.border}`,
                    paddingBottom: theme.spacing.sm,
                    paddingTop: theme.spacing.sm,
                  }}
                >
                  <Button
                    type='link'
                    css={{
                      textAlign: 'left',
                      overflowX: 'auto',
                      overflowY: 'hidden',
                    }}
                    onClick={() => {
                      setSelectedDatasetWithRun({ datasetWithTags: dataset, runData: runData });
                      setIsOpen(true);
                    }}
                  >
                    <ExperimentViewDatasetWithContext
                      datasetWithTags={dataset}
                      displayTextAsLink={false}
                    />
                  </Button>
                </div>
              ))}
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
                      <Tooltip title={datasetWithTags.dataset.name}>
                        <Typography.Title
                          ellipsis
                          level={3}
                          css={{ marginBottom: 0, maxWidth: 200 }}
                        >
                          {datasetWithTags.dataset.name}
                        </Typography.Title>
                      </Tooltip>
                      {contextTag && (
                        <Tag
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
                  color='secondary'
                  css={{ marginBottom: 0 }}
                  title={fullProfile}
                >
                  {datasetWithTags.dataset.digest},{' '}
                  {datasetWithTags.dataset.profile && datasetWithTags.dataset.profile !== 'null' ? (
                    datasetWithTags.dataset.profile.length > MAX_PROFILE_LENGTH ? (
                      `${datasetWithTags.dataset.profile.substring(0, MAX_PROFILE_LENGTH)} ...`
                    ) : (
                      datasetWithTags.dataset.profile
                    )
                  ) : (
                    <FormattedMessage
                      defaultMessage='No profile available'
                      description='Text for no profile available in the experiment run dataset drawer'
                    />
                  )}
                </Typography.Title>
              </div>
              <ExperimentViewDatasetLink datasetWithTags={datasetWithTags} runTags={runData.tags} />
            </div>
            {/* dataset schema */}
            <Divider css={{ marginTop: theme.spacing.xs, marginBottom: theme.spacing.xs }} />
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
  colorPill: {
    width: 12,
    height: 12,
    borderRadius: 6,
    flexShrink: 0,
    // Straighten it up on retina-like screens
    '@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi)': {
      marginBottom: 1,
    },
  },
};
