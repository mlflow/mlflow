import { useCallback, useState } from 'react';

import {
  Button,
  FullscreenExitIcon,
  FullscreenIcon,
  GearIcon,
  Popover,
  SegmentedControlButton,
  SegmentedControlGroup,
  Switch,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  type RadioChangeEvent,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { GraphOrientation } from './GraphView.types';

interface GraphViewFloatingToolbarProps {
  orientation: GraphOrientation;
  onOrientationChange: (orientation: GraphOrientation) => void;
  showMinimap: boolean;
  onShowMinimapChange: (show: boolean) => void;
  showStepSequence: boolean;
  onShowStepSequenceChange: (show: boolean) => void;
  isGraphExpanded: boolean;
  onToggleGraphExpand: () => void;
  onFitView: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}

export const GraphViewFloatingToolbar = ({
  orientation,
  onOrientationChange,
  showMinimap,
  onShowMinimapChange,
  showStepSequence,
  onShowStepSequenceChange,
  isGraphExpanded,
  onToggleGraphExpand,
  onFitView,
  onZoomIn,
  onZoomOut,
}: GraphViewFloatingToolbarProps) => {
  const { theme } = useDesignSystemTheme();
  const [settingsOpen, setSettingsOpen] = useState(false);

  const handleOrientationChange = useCallback(
    (e: RadioChangeEvent) => {
      onOrientationChange(e.target.value as GraphOrientation);
    },
    [onOrientationChange],
  );

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
          icon={<FitViewIcon />}
          size="small"
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
          icon={<PlusIcon />}
          size="small"
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
          icon={<MinusIcon />}
          size="small"
          onClick={onZoomOut}
          css={toolbarButtonStyle(theme)}
        />
      </Tooltip>

      <Popover.Root
        componentId="graph-view-toolbar.settings-popover"
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
      >
        <Popover.Trigger asChild>
          <div>
            <Tooltip
              componentId="graph-view-toolbar.settings"
              content={
                <FormattedMessage
                  defaultMessage="More options"
                  description="Tooltip for the settings button in the graph toolbar"
                />
              }
              side="left"
            >
              <Button
                componentId="graph-view-toolbar.settings-button"
                icon={<GearIcon />}
                size="small"
                css={toolbarButtonStyle(theme)}
              />
            </Tooltip>
          </div>
        </Popover.Trigger>
        <Popover.Content align="end" side="left">
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, minWidth: 180 }}>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <Typography.Text size="sm" bold>
                <FormattedMessage
                  defaultMessage="DAG Orientation"
                  description="Label for the DAG orientation toggle in the graph settings"
                />
              </Typography.Text>
              <SegmentedControlGroup
                componentId="graph-view-toolbar.orientation-control"
                name="graph-orientation"
                value={orientation}
                onChange={handleOrientationChange}
              >
                <SegmentedControlButton value="TB">
                  <span css={{ fontSize: 14, lineHeight: '16px' }}>↓</span>
                </SegmentedControlButton>
                <SegmentedControlButton value="LR">
                  <span css={{ fontSize: 14, lineHeight: '16px' }}>→</span>
                </SegmentedControlButton>
              </SegmentedControlGroup>
            </div>
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: theme.spacing.sm,
              }}
            >
              <Typography.Text size="sm" bold>
                <FormattedMessage
                  defaultMessage="Show minimap"
                  description="Label for the minimap toggle in the graph settings"
                />
              </Typography.Text>
              <Switch
                componentId="graph-view-toolbar.minimap-toggle"
                checked={showMinimap}
                onChange={onShowMinimapChange}
              />
            </div>
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: theme.spacing.sm,
              }}
            >
              <Typography.Text size="sm" bold>
                <FormattedMessage
                  defaultMessage="Show step numbers"
                  description="Label for the step sequence toggle in the graph settings"
                />
              </Typography.Text>
              <Switch
                componentId="graph-view-toolbar.step-sequence-toggle"
                checked={showStepSequence}
                onChange={onShowStepSequenceChange}
              />
            </div>
          </div>
        </Popover.Content>
      </Popover.Root>
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

function FitViewIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M3 2H6V3H3.5C3.22386 3 3 3.22386 3 3.5V6H2V3C2 2.44772 2.44772 2 3 2Z" fill="currentColor" />
      <path d="M10 2H13C13.5523 2 14 2.44772 14 3V6H13V3.5C13 3.22386 12.7761 3 12.5 3H10V2Z" fill="currentColor" />
      <path d="M3 10V12.5C3 12.7761 3.22386 13 3.5 13H6V14H3C2.44772 14 2 13.5523 2 13V10H3Z" fill="currentColor" />
      <path
        d="M13 10V12.5C13 12.7761 12.7761 13 12.5 13H10V14H13C13.5523 14 14 13.5523 14 13V10H13Z"
        fill="currentColor"
      />
      <rect x="5" y="5" width="6" height="6" rx="0.5" stroke="currentColor" fill="none" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M8 3V13M3 8H13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function MinusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M3 8H13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}
