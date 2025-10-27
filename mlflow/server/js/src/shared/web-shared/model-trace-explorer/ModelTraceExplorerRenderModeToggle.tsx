import {
  MIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  TextBoxIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

export function ModelTraceExplorerRenderModeToggle({
  shouldRenderMarkdown,
  setShouldRenderMarkdown,
}: {
  shouldRenderMarkdown: boolean;
  setShouldRenderMarkdown: (value: boolean) => void;
}) {
  const { theme } = useDesignSystemTheme();

  return (
    <SegmentedControlGroup
      data-testid="model-trace-explorer-render-mode-toggle"
      name="render-mode"
      size="small"
      componentId={`shared.model-trace-explorer.toggle-markdown-rendering-${!shouldRenderMarkdown}`}
      value={shouldRenderMarkdown}
      onChange={(event) => {
        setShouldRenderMarkdown(event.target.value);
      }}
    >
      <SegmentedControlButton data-testid="model-trace-explorer-render-raw-input-button" value={false}>
        <Tooltip
          componentId="shared.model-trace-explorer.raw-input-rendering-tooltip"
          content={
            <FormattedMessage
              defaultMessage="Raw input"
              description="Tooltip content for a button that changes the render mode of the data to raw input (JSON)"
            />
          }
        >
          <div css={{ display: 'flex', alignItems: 'center' }}>
            <TextBoxIcon css={{ fontSize: theme.typography.fontSizeLg }} />
          </div>
        </Tooltip>
      </SegmentedControlButton>
      <SegmentedControlButton data-testid="model-trace-explorer-render-default-button" value>
        <Tooltip
          componentId="shared.model-trace-explorer.default-rendering-tooltip"
          content={
            <FormattedMessage
              defaultMessage="Default rendering"
              description="Tooltip content for a button that changes the render mode to default"
            />
          }
        >
          <div css={{ display: 'flex', alignItems: 'center' }}>
            <MIcon css={{ fontSize: theme.typography.fontSizeLg }} />
          </div>
        </Tooltip>
      </SegmentedControlButton>
    </SegmentedControlGroup>
  );
}
