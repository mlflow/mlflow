import {
  Button,
  ColumnsIcon,
  DropdownMenu,
  Notification,
  RowsIcon,
  Typography,
  useDesignSystemTheme,
  Tag,
  Tooltip,
  Overflow,
} from '@databricks/design-system';
import { ColumnDef } from '@tanstack/react-table';
import { useCallback, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { EvaluationDataset, EvaluationDatasetRecord } from '../types';
import { isUserFacingTag, parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';

const BASE_NOTIFICATION_COMPONENT_ID = 'mlflow.eval-datasets.records-toolbar.notification';

const getTotalRecordsCount = (profile: string | undefined): number | undefined => {
  if (!profile) {
    return undefined;
  }

  const profileJson = parseJSONSafe(profile);
  return profileJson?.num_records ?? undefined;
};

export const ExperimentEvaluationDatasetRecordsToolbar = ({
  dataset,
  datasetRecords,
  columns,
  columnVisibility,
  setColumnVisibility,
  rowSize,
  setRowSize,
}: {
  dataset: EvaluationDataset;
  datasetRecords: EvaluationDatasetRecord[];
  columns: ColumnDef<EvaluationDatasetRecord, any>[];
  columnVisibility: Record<string, boolean>;
  setColumnVisibility: (columnVisibility: Record<string, boolean>) => void;
  rowSize: 'sm' | 'md' | 'lg';
  setRowSize: (rowSize: 'sm' | 'md' | 'lg') => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [showNotification, setShowNotification] = useState(false);

  const datasetName = dataset?.name;
  const profile = dataset?.profile;
  const datasetId = dataset?.dataset_id;
  const parsedTags = (dataset?.tags && parseJSONSafe(dataset.tags)) || {};
  const userFacingTags = Object.entries(parsedTags as Record<string, any>)
    .filter(([key]) => isUserFacingTag(key))
    .map(([key, value]) => ({ key, value }));
  const totalRecordsCount = getTotalRecordsCount(profile);
  const loadedRecordsCount = datasetRecords.length;

  const handleCopy = useCallback(() => {
    setShowNotification(true);
    setTimeout(() => setShowNotification(false), 2000);
  }, []);

  return (
    <>
      <div
        css={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          marginBottom: theme.spacing.sm,
          paddingTop: theme.spacing.sm,
          paddingBottom: theme.spacing.sm,
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            paddingLeft: theme.spacing.sm,
            paddingRight: theme.spacing.sm,
            gap: theme.spacing.xs,
          }}
        >
          <Typography.Title level={3} withoutMargins>
            {datasetName}
          </Typography.Title>
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              flexWrap: 'wrap',
              marginTop: theme.spacing.xs,
              marginBottom: theme.spacing.xs,
            }}
          >
            {' '}
            {/* ID pill (copy on click) */}
            {datasetId && (
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <Typography.Text color="secondary" size="sm">
                  <FormattedMessage defaultMessage="ID" description="Label for the dataset id pill" />
                </Typography.Text>
                <Tooltip componentId="mlflow.eval-datasets.records-toolbar.dataset-id.tooltip" content={datasetId}>
                  <Tag
                    componentId="mlflow.eval-datasets.records-toolbar.dataset-id"
                    color="indigo"
                    css={{ cursor: 'pointer' }}
                    onClick={() => {
                      navigator.clipboard.writeText(datasetId);
                      handleCopy();
                    }}
                  >
                    <Typography.Text css={{ fontFamily: 'monospace' }}>{datasetId}</Typography.Text>
                  </Tag>
                </Tooltip>
              </div>
            )}
            {/* Tags pills */}
            {userFacingTags.length > 0 && (
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <Typography.Text color="secondary" size="sm">
                  <FormattedMessage defaultMessage="Tags" description="Label for the dataset tags pills" />
                </Typography.Text>
                <Overflow noMargin>
                  {userFacingTags.map(({ key, value }) => {
                    const fullText = `${key}: ${String(value)}`;
                    const compId = `mlflow.eval-datasets.records-toolbar.dataset-tag.${key}`;
                    return (
                      <Tooltip key={compId} componentId={compId} content={fullText}>
                        <Tag
                          componentId={compId}
                          color="teal"
                          css={{ cursor: 'pointer' }}
                          onClick={() => {
                            navigator.clipboard.writeText(fullText);
                            handleCopy();
                          }}
                        >
                          {fullText}
                        </Tag>
                      </Tooltip>
                    );
                  })}
                </Overflow>
              </div>
            )}
          </div>
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage
              defaultMessage="Displaying {loadedRecordsCount} of {totalRecordsCount, plural, =1 {1 record} other {# records}}"
              description="Label for the number of records displayed"
              values={{ loadedRecordsCount: loadedRecordsCount ?? 0, totalRecordsCount: totalRecordsCount ?? 0 }}
            />
          </Typography.Text>
        </div>
        <div css={{ display: 'flex', alignItems: 'flex-start' }}>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button componentId="mlflow.eval-datasets.records-toolbar.row-size-toggle" icon={<RowsIcon />} />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="end">
              <DropdownMenu.RadioGroup
                componentId="mlflow.eval-datasets.records-toolbar.row-size-radio"
                value={rowSize}
                onValueChange={(value) => setRowSize(value as 'sm' | 'md' | 'lg')}
              >
                <DropdownMenu.Label>
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Row height" description="Label for the row height radio group" />
                  </Typography.Text>
                </DropdownMenu.Label>
                <DropdownMenu.RadioItem key="sm" value="sm">
                  <DropdownMenu.ItemIndicator />
                  <Typography.Text>
                    <FormattedMessage defaultMessage="Small" description="Small row size" />
                  </Typography.Text>
                </DropdownMenu.RadioItem>
                <DropdownMenu.RadioItem key="md" value="md">
                  <DropdownMenu.ItemIndicator />
                  <Typography.Text>
                    <FormattedMessage defaultMessage="Medium" description="Medium row size" />
                  </Typography.Text>
                </DropdownMenu.RadioItem>
                <DropdownMenu.RadioItem key="lg" value="lg">
                  <DropdownMenu.ItemIndicator />
                  <Typography.Text>
                    <FormattedMessage defaultMessage="Large" description="Large row size" />
                  </Typography.Text>
                </DropdownMenu.RadioItem>
              </DropdownMenu.RadioGroup>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button componentId="mlflow.eval-datasets.records-toolbar.columns-toggle" icon={<ColumnsIcon />} />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content>
              {columns.map((column) => (
                <DropdownMenu.CheckboxItem
                  componentId="YOUR_TRACKING_ID"
                  key={column.id}
                  checked={columnVisibility[column.id ?? ''] ?? false}
                  onCheckedChange={(checked) =>
                    setColumnVisibility({
                      ...columnVisibility,
                      [column.id ?? '']: checked,
                    })
                  }
                >
                  <DropdownMenu.ItemIndicator />
                  <Typography.Text>{column.header}</Typography.Text>
                </DropdownMenu.CheckboxItem>
              ))}
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
      </div>

      {showNotification && (
        <Notification.Provider>
          <Notification.Root severity="success" componentId={BASE_NOTIFICATION_COMPONENT_ID}>
            <Notification.Title>
              <FormattedMessage
                defaultMessage="Copied to clipboard"
                description="Success message for the notification"
              />
            </Notification.Title>
          </Notification.Root>
          <Notification.Viewport />
        </Notification.Provider>
      )}
    </>
  );
};
