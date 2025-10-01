import {
  Button,
  ColumnsIcon,
  DropdownMenu,
  RowsIcon,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { isNil } from 'lodash';
import { FormattedMessage } from 'react-intl';

export const ExperimentEvaluationDatasetRecordsToolbar = ({
  datasetName,
  loadedRecordsCount,
  totalRecordsCount,
  rowSize,
  setRowSize,
}: {
  datasetName: string;
  loadedRecordsCount?: number;
  totalRecordsCount?: number;
  rowSize: 'sm' | 'md' | 'lg';
  setRowSize: (rowSize: 'sm' | 'md' | 'lg') => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        marginBottom: theme.spacing.sm + 2,
      }}
    >
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          paddingLeft: theme.spacing.sm,
          paddingRight: theme.spacing.sm,
        }}
      >
        <Typography.Title level={3} withoutMargins>
          {datasetName}
        </Typography.Title>
        {!isNil(loadedRecordsCount) && !isNil(totalRecordsCount) && (
          <Typography.Text color="secondary" size="sm">
            Displaying {loadedRecordsCount} of {totalRecordsCount} records
          </Typography.Text>
        )}
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
        <Button componentId="mlflow.eval-datasets.records-toolbar.columns-toggle" icon={<ColumnsIcon />} />
      </div>
    </div>
  );
};
