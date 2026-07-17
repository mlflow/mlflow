import {
  Button,
  FullscreenExitIcon,
  FullscreenIcon,
  Tooltip,
  ZoomInIcon,
  ZoomOutIcon,
  ZoomToFitIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

interface GraphViewFloatingToolbarProps {
  isGraphExpanded: boolean;
  onToggleGraphExpand: () => void;
  onFitView: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}

export const GraphViewFloatingToolbar = ({
  isGraphExpanded,
  onToggleGraphExpand,
  onFitView,
  onZoomIn,
  onZoomOut,
}: GraphViewFloatingToolbarProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
        borderRadius: theme.borders.borderRadiusMd,
        overflow: 'hidden',
        boxShadow: theme.shadows.md,
        border: `1px solid ${theme.colors.border}`,
      }}
    >
      <Tooltip
        componentId="graph-view-toolbar.expand"
        content={
          isGraphExpanded ? (
            <FormattedMessage
              defaultMessage="Collapse graph"
              description="Tooltip for the button that collapses the graph view"
            />
          ) : (
            <FormattedMessage
              defaultMessage="Expand graph"
              description="Tooltip for the button that expands the graph view"
            />
          )
        }
        side="left"
      >
        <Button
          componentId="graph-view-toolbar.expand-button"
          icon={isGraphExpanded ? <FullscreenExitIcon /> : <FullscreenIcon />}
          size="small"
          aria-label={isGraphExpanded ? 'Collapse graph' : 'Expand graph'}
          onClick={onToggleGraphExpand}
          css={toolbarButtonStyle(theme)}
        />
      </Tooltip>
      <Tooltip
        componentId="graph-view-toolbar.fit-view"
        content={
          <FormattedMessage
            defaultMessage="Fit to view"
            description="Tooltip for the button that fits the graph to the viewport"
          />
        }
        side="left"
      >
        <Button
          componentId="graph-view-toolbar.fit-view-button"
          icon={<ZoomToFitIcon />}
          size="small"
          aria-label="Fit to view"
          onClick={onFitView}
          css={toolbarButtonStyle(theme)}
        />
      </Tooltip>
      <Tooltip
        componentId="graph-view-toolbar.zoom-in"
        content={
          <FormattedMessage defaultMessage="Zoom in" description="Tooltip for the button that zooms in on the graph" />
        }
        side="left"
      >
        <Button
          componentId="graph-view-toolbar.zoom-in-button"
          icon={<ZoomInIcon />}
          size="small"
          aria-label="Zoom in"
          onClick={onZoomIn}
          css={toolbarButtonStyle(theme)}
        />
      </Tooltip>
      <Tooltip
        componentId="graph-view-toolbar.zoom-out"
        content={
          <FormattedMessage
            defaultMessage="Zoom out"
            description="Tooltip for the button that zooms out of the graph"
          />
        }
        side="left"
      >
        <Button
          componentId="graph-view-toolbar.zoom-out-button"
          icon={<ZoomOutIcon />}
          size="small"
          aria-label="Zoom out"
          onClick={onZoomOut}
          css={toolbarButtonStyle(theme)}
        />
      </Tooltip>
    </div>
  );
};

const toolbarButtonStyle = (theme: ReturnType<typeof useDesignSystemTheme>['theme']) => ({
  borderRadius: 0,
  border: 'none',
  backgroundColor: theme.colors.backgroundPrimary,
  color: theme.colors.textSecondary,
  '&:hover': {
    backgroundColor: theme.colors.actionTertiaryBackgroundHover,
  },
});
