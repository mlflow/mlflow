import { useEffect, useMemo, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  SparkleFillIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { useAssistant } from './AssistantContext';
import { useFloatingObstructionWidth } from './useFloatingObstruction';
import { useLogTelemetryEvent } from '../telemetry/hooks/useLogTelemetryEvent';

// Width the pill expands to on hover; also used to keep it on-screen when shifted left.
const FAB_EXPANDED_MAX_WIDTH = 240;

// Tracks the viewport width reactively so the "does the button still fit beside the
// drawer?" check re-runs when the window is resized (React won't re-render on resize alone).
const useWindowWidth = (): number => {
  const [width, setWidth] = useState(() => window.innerWidth);
  useEffect(() => {
    let frame = 0;
    // Coalesce the burst of resize events into one update per animation frame; React
    // then bails out of re-rendering when innerWidth hasn't actually changed.
    const onResize = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => setWidth(window.innerWidth));
    };
    window.addEventListener('resize', onResize);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', onResize);
    };
  }, []);
  return width;
};

/**
 * Global floating entry point for the Assistant. Pinned bottom-right on every
 * page (local server only), mirroring the convention used by other observability
 * tools. Opens the existing Assistant side panel. Hidden while the panel is open
 * to avoid overlapping it.
 *
 * On first load it opens the panel once (a one-time discovery boost), persisted so
 * it never auto-opens again.
 */
export const AssistantFloatingButton = () => {
  const { theme } = useDesignSystemTheme();
  const { isLocalServer, isPanelOpen, openPanel } = useAssistant();
  const obstructionWidth = useFloatingObstructionWidth();
  const windowWidth = useWindowWidth();
  const logTelemetryEvent = useLogTelemetryEvent();
  const viewId = useMemo(() => uuidv4(), []);

  // One-time first-load auto-open, persisted so the panel opens itself only once.
  const [autoOpened, setAutoOpened] = useLocalStorage({
    key: 'mlflow.assistant.fab.autoOpenedOnFirstLoad',
    version: 1,
    initialValue: false,
  });

  // On first load, open the panel once to surface the assistant. Mark it done even if the
  // panel is already open (e.g. restored from a prior session) so a later reload won't force it.
  useEffect(() => {
    if (!isLocalServer || autoOpened) {
      return;
    }
    setAutoOpened(true);
    if (!isPanelOpen) {
      openPanel();
    }
  }, [isLocalServer, isPanelOpen, autoOpened, setAutoOpened, openPanel]);

  // Hide only when the assistant is unavailable or already open. When a right-side
  // surface is open it doesn't hide the button — it reports the width it reserves (via
  // useRegisterFloatingObstruction) and the button shifts to its left so it stays
  // visible without overlapping. This is generic across any registering surface.
  if (!isLocalServer || isPanelOpen) {
    return null;
  }

  const baseInset = theme.spacing.md;
  // Sit just to the left of an open right-side surface. If that surface is so wide the
  // (expandable) button can't sit clear of it, yield entirely rather than float on top —
  // the surface has its own assistant entry (e.g. the trace drawer's header button).
  const rightInset = obstructionWidth > 0 ? obstructionWidth + theme.spacing.md : baseInset;
  const fitsClearOfObstruction = rightInset + FAB_EXPANDED_MAX_WIDTH + theme.spacing.md <= windowWidth;
  // When the surface is too wide for the button to sit clear, fade it out (kept mounted so
  // the transition can play) rather than overlapping the surface.
  const hiddenByObstruction = obstructionWidth > 0 && !fitsClearOfObstruction;

  const handleOpen = () => {
    openPanel();
    logTelemetryEvent({
      componentId: 'mlflow.assistant.fab',
      componentViewId: viewId,
      componentType: DesignSystemEventProviderComponentTypes.Button,
      componentSubType: null,
      eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
    });
  };

  // Standard DuBois control height, so the bubble stays in step with other UI controls.
  const fabSize = theme.general.heightBase;
  const iconSize = theme.typography.fontSizeXl;
  // Inverted high-contrast surface: near-black in light mode, white in dark mode.
  const bubbleBackground = theme.isDarkMode ? theme.colors.white : theme.colors.grey800;
  const labelColor = theme.isDarkMode ? theme.colors.grey800 : theme.colors.white;

  // Plain button (not the DuBois Button) so we fully control the background and the
  // hover-to-expand pill behaviour without fighting the component's own styles.
  const fabButton = (
    <button
      type="button"
      data-assistant-ui="true"
      onClick={handleOpen}
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        height: fabSize,
        minWidth: fabSize,
        maxWidth: fabSize,
        paddingInline: (fabSize - iconSize) / 2,
        paddingBlock: 0,
        border: 'none',
        borderRadius: fabSize / 2,
        cursor: 'pointer',
        appearance: 'none',
        background: bubbleBackground,
        color: labelColor,
        boxShadow: theme.general.shadowHigh,
        overflow: 'hidden',
        whiteSpace: 'nowrap',
        '@media (prefers-reduced-motion: no-preference)': {
          transition: 'max-width 0.25s ease',
        },
        // Expand into a pill that reveals the label on hover/focus.
        '&:hover, &:focus-visible': {
          maxWidth: FAB_EXPANDED_MAX_WIDTH,
        },
        '& .assistant-fab-label': {
          opacity: 0,
          marginLeft: 0,
          fontWeight: theme.typography.typographyBoldFontWeight,
          fontSize: theme.typography.fontSizeMd,
          '@media (prefers-reduced-motion: no-preference)': {
            transition: 'opacity 0.2s ease, margin 0.2s ease',
          },
        },
        '&:hover .assistant-fab-label, &:focus-visible .assistant-fab-label': {
          opacity: 1,
          marginLeft: theme.spacing.sm,
        },
      }}
    >
      <SparkleFillIcon color="ai" css={{ fontSize: iconSize, flexShrink: 0 }} />
      <span className="assistant-fab-label">
        <FormattedMessage
          defaultMessage="MLflow Assistant"
          description="Label revealed when hovering the floating MLflow assistant button"
        />
      </span>
    </button>
  );

  return (
    // data-assistant-ui marks this as part of the assistant UI so that
    // AssistantAwareDrawer does not treat clicks here as outside clicks.
    // See AssistantAwareDrawer.tsx.
    <div
      data-assistant-ui="true"
      css={{
        position: 'fixed',
        bottom: baseInset,
        // Sits just left of any open right-side surface (e.g. a drawer) so it stays
        // visible without overlapping; otherwise rests in the bottom-right corner.
        right: rightInset,
        // Above the drawer (design system drawer content sits at zIndexBase + 2/3).
        zIndex: theme.options.zIndexBase + 10,
        display: 'flex',
        opacity: hiddenByObstruction ? 0 : 1,
        // visibility (not unmount) so it drops out of focus/AT order once faded.
        visibility: hiddenByObstruction ? 'hidden' : 'visible',
        pointerEvents: hiddenByObstruction ? 'none' : 'auto',
        '@media (prefers-reduced-motion: no-preference)': {
          transition: hiddenByObstruction ? 'opacity 0.15s ease, visibility 0s 0.15s' : 'opacity 0.15s ease',
        },
      }}
    >
      {fabButton}
    </div>
  );
};
