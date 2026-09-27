import { Button, LinkIcon, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { useCopyController } from '../../../copy/useCopyController';
import { ModelIconType, type ModelTraceToolCall } from '../ModelTrace.types';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';

const formatArgumentValue = (value: unknown): string => {
  if (typeof value === 'string') {
    return value;
  }
  if (value === null || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const getToolCallArguments = (rawArguments: string): Array<[string, string]> => {
  if (!rawArguments) {
    return [];
  }

  try {
    const parsedArguments = JSON.parse(rawArguments);
    if (parsedArguments && typeof parsedArguments === 'object' && !Array.isArray(parsedArguments)) {
      return Object.entries(parsedArguments).map(([key, value]) => [key, formatArgumentValue(value)]);
    }

    return [['arguments', formatArgumentValue(parsedArguments)]];
  } catch {
    return [['arguments', rawArguments]];
  }
};

export function ModelTraceExplorerToolIcon(): React.ReactElement | null {
  return <ModelTraceExplorerIcon type={ModelIconType.WRENCH} />;
}

export function ModelTraceExplorerToolCallIdLink({ toolCallId }: { toolCallId: string }): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const copyTooltip = intl.formatMessage({
    defaultMessage: 'Copy tool call ID',
    description: 'Tooltip for copying a tool call ID in the trace chat renderer',
  });
  const { ariaLabel, copy, handleTooltipOpenChange, tooltipMessage, tooltipOpen } = useCopyController(
    toolCallId,
    copyTooltip,
  );

  return (
    <Tooltip
      componentId="shared.model-trace-explorer.tool-call-id-tooltip"
      content={tooltipMessage}
      onOpenChange={handleTooltipOpenChange}
      open={tooltipOpen}
    >
      <Button
        aria-label={ariaLabel}
        componentId="shared.model-trace-explorer.copy-tool-call-id"
        icon={<LinkIcon />}
        onClick={(event) => {
          event.stopPropagation();
          copy();
        }}
        size="small"
        type="tertiary"
        css={{
          alignSelf: 'center',
          width: theme.spacing.md,
          minWidth: theme.spacing.md,
          height: theme.spacing.md,
          padding: 0,
          color: theme.colors.textSecondary,
          flexShrink: 0,
          opacity: 0.75,
          verticalAlign: 'middle',
          svg: {
            width: theme.typography.fontSizeSm,
            height: theme.typography.fontSizeSm,
          },
          '&:hover': {
            opacity: 1,
          },
        }}
      />
    </Tooltip>
  );
}

export function ModelTraceExplorerToolCallMessage({
  toolCall,
}: {
  toolCall: ModelTraceToolCall;
}): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();
  const argumentRows = getToolCallArguments(toolCall.function.arguments);

  return (
    <div key={toolCall.id} css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          gap: theme.spacing.sm,
          padding: `0 ${theme.spacing.sm + theme.spacing.xs}px`,
        }}
      >
        <ModelTraceExplorerToolIcon />
        <Typography.Text bold css={{ whiteSpace: 'nowrap' }}>
          {toolCall.function.name}
        </Typography.Text>
        <ModelTraceExplorerToolCallIdLink toolCallId={toolCall.id} />
      </div>
      {argumentRows.length > 0 && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
            paddingLeft: theme.spacing.sm + theme.spacing.xs + theme.general.iconSize + theme.spacing.sm,
            paddingRight: theme.spacing.sm + theme.spacing.xs,
          }}
        >
          {argumentRows.map(([key, value]) => (
            <div key={key} css={{ display: 'flex', alignItems: 'baseline', gap: theme.spacing.md, minWidth: 0 }}>
              <span
                css={{
                  width: theme.spacing.xs,
                  height: theme.spacing.xs,
                  borderRadius: theme.borders.borderRadiusFull,
                  backgroundColor: theme.colors.backgroundSecondary,
                  flexShrink: 0,
                }}
              />
              <Typography.Text color="secondary" bold css={{ minWidth: theme.spacing.xl * 3, whiteSpace: 'nowrap' }}>
                {key}
              </Typography.Text>
              <Typography.Text css={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', minWidth: 0 }}>
                {value}
              </Typography.Text>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
