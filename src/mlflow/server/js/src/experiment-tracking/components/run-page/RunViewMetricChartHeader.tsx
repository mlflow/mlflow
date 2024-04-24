import {
  Button,
  DragIcon,
  DropdownMenu,
  OverflowIcon,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ConnectDragSource } from 'react-dnd';
import { FormattedMessage } from 'react-intl';

/**
 * Header for a single run view chart box
 */
export const RunViewMetricChartHeader = ({
  dragHandleRef,
  metricKey,
  canMoveUp,
  canMoveDown,
  onMoveUp,
  onMoveDown,
  onRefresh,
  isRefreshing = false,
}: {
  dragHandleRef: ConnectDragSource;
  metricKey: string;
  canMoveUp: boolean;
  canMoveDown: boolean;
  onMoveUp: () => void;
  onMoveDown: () => void;
  onRefresh: () => void;
  isRefreshing?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        gap: theme.spacing.sm,
        justifyContent: 'space-between',
        marginBottom: theme.spacing.sm,
        alignItems: 'center',
      }}
    >
      <div
        css={{
          display: 'flex',
          gap: theme.spacing.sm,
          alignItems: 'center',
          overflow: 'hidden',
        }}
      >
        <DragIcon
          css={{
            cursor: 'grab',
          }}
          ref={dragHandleRef}
        />
        <Typography.Title level={4} ellipsis withoutMargins>
          {metricKey}
        </Typography.Title>
      </div>
      <div css={{ flex: 1 }} />
      {isRefreshing && (
        <Spinner
          label={
            <FormattedMessage
              defaultMessage="Chart data is refreshing"
              description="Run page > Charts tab > Chart box header > Refreshing chart accessible label"
            />
          }
        />
      )}
      <DropdownMenu.Root modal={false}>
        <DropdownMenu.Trigger asChild>
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewmetricchartheader.tsx_78"
            css={{ flexShrink: 0 }}
            icon={<OverflowIcon />}
            size="small"
            title="Chart options"
          />
        </DropdownMenu.Trigger>
        <DropdownMenu.Content>
          <DropdownMenu.Item disabled={!canMoveUp} onClick={onMoveUp}>
            <FormattedMessage
              defaultMessage="Move up"
              description="Run page > Charts tab > Chart box header > Move up dropdown option"
            />
          </DropdownMenu.Item>
          <DropdownMenu.Item disabled={!canMoveDown} onClick={onMoveDown}>
            <FormattedMessage
              defaultMessage="Move down"
              description="Run page > Charts tab > Chart box header > Move down dropdown option"
            />
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    </div>
  );
};
