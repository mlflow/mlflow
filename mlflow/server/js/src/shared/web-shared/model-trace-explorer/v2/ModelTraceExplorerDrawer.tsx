import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import {
  ApplyDesignSystemContextOverrides,
  Button,
  CheckCircleIcon,
  CheckboxIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CloseIcon,
  ClockIcon,
  DatabaseIcon,
  FullscreenExitIcon,
  FullscreenIcon,
  LinkIcon,
  PlusIcon,
  SearchIcon,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  XCircleIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { Global, keyframes } from '@emotion/react';
import { useAssistant } from '@mlflow/mlflow/src/assistant';

import { shouldEnableModelTraceExplorerCustomTraceView } from '../FeatureUtils';
import { ModelTraceExplorerCustomViewSelector } from './ModelTraceExplorerCustomViewSelector';
import { isTraceCostType } from '../ModelTraceExplorerCostHoverCard';
import { ModelTraceExplorerAssistantButton } from './ModelTraceExplorerAssistantButton';
import { ModelTraceExplorerSkeleton } from '../ModelTraceExplorerSkeleton';
import {
  type ModelTraceExplorerDisplayMode,
  ModelTraceExplorerRightPaneHeaderActionsProvider,
  useModelTraceExplorerContext,
} from './ModelTraceExplorerContext';
import { useCustomViewAssistantConnector } from '../custom-view/assistant/CustomViewAssistantConnector';
import { useOptionalCustomViewDefinition } from '../custom-view/CustomViewDefinitionContext';
import type { ModelTraceInfoV3 } from './ModelTrace.types';
import { getTraceCost, getTraceTokenUsage } from './ModelTraceExplorer.utils';
import { CostMetadataItem, TokenUsageMetadataItem } from './right-pane/ModelTraceExplorerRightPaneHeader';
import { useCopyController } from '../../copy/useCopyController';
import { useLocation } from '../RoutingUtils';

const CUSTOM_VIEW_INITIAL_WIDTH = '50vw';
const MIN_DRAWER_WIDTH = 480;
const MAX_DRAWER_WIDTH_RATIO = 0.9;
const DRAWER_CLOSE_ANIMATION_MS = 180;
const TRACE_METADATA_MIN_WIDTH = 960;
const LABELED_PRIMARY_ACTIONS_MIN_WIDTH = 800;
// Targets the design system Drawer.Content node (it renders these data attributes from the
// Content componentId below), used by the Global rules that reproduce managed's header layout.
const DRAWER_CONTENT_SELECTOR =
  '[data-component-type="drawer_content"][data-component-id="mlflow.evaluations_review.modal"]';
const headerIconCss = { svg: { width: 15, height: 15 } };
const statusIconCss = { svg: { width: 12, height: 12 } };

const formatHeaderDuration = (value: string): string => {
  const match = value.match(/^(-?\d+(?:\.\d+)?)(.*)$/);
  if (!match) {
    return value;
  }

  const numericValue = Number(match[1]);
  if (!Number.isFinite(numericValue)) {
    return value;
  }

  return match[2] === 's' ? `${(numericValue * 1000).toFixed(2)}ms` : `${numericValue.toFixed(2)}${match[2]}`;
};

const formatHeaderCost = (cost: number): string => `$${cost.toFixed(2)}`;

const resolveWidthToPixels = (width: number | string, viewportWidth = window.innerWidth): number => {
  if (typeof width === 'number') {
    return width;
  }
  if (width.endsWith('vw')) {
    return (parseFloat(width) / 100) * viewportWidth;
  }
  if (width.endsWith('px')) {
    return parseFloat(width);
  }
  return MIN_DRAWER_WIDTH;
};

const drawerSlideOutAnimation = keyframes({
  '0%': {
    transform: 'translate(0, 0)',
  },
  '100%': {
    transform: 'translate(100%, 0)',
  },
});

export interface ModelTraceExplorerDrawerProps {
  children: React.ReactNode;
  selectPreviousEval: () => void;
  selectNextEval: () => void;
  isPreviousAvailable: boolean;
  isNextAvailable: boolean;
  handleClose: () => void;
  isLoading?: boolean;
  experimentId?: string;
  traceInfo?: ModelTraceInfoV3;
  renderManagedAddToDatasetDropdown?: (params: {
    children: React.ReactNode;
    onOpenChange: (open: boolean) => void;
    open: boolean;
  }) => React.ReactNode;
}

export const ModelTraceExplorerDrawer = ({
  selectPreviousEval,
  selectNextEval,
  isPreviousAvailable,
  isNextAvailable,
  handleClose,
  children,
  isLoading,
  experimentId,
  traceInfo,
  renderManagedAddToDatasetDropdown,
}: ModelTraceExplorerDrawerProps): JSX.Element => {
  const { getPrefixedClassName, theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { openAssistant: openCustomViewAssistant } = useCustomViewAssistantConnector();
  const customViewDefinition = useOptionalCustomViewDefinition();
  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isSearchVisible, setSearchVisible] = useState(false);
  const [isFlagForReviewDropdownOpen, setIsFlagForReviewDropdownOpen] = useState(false);
  const [isFlagForReviewTooltipOpen, setIsFlagForReviewTooltipOpen] = useState(false);
  const fullscreenActionButtonCss = isFullscreen
    ? {
        '&&, &&:hover, &&:active': {
          borderColor: 'transparent !important',
        },
        '&& > .anticon + span': {
          marginLeft: `${theme.spacing.sm}px !important`,
        },
      }
    : undefined;
  const closeTimerRef = useRef<number | undefined>(undefined);
  const isClosingRef = useRef(false);
  const {
    renderExportTracesToDatasetsModal,
    renderAddToReviewQueueDropdown,
    DrawerComponent,
    drawerWidth = '90vw',
    isAssistantPanelOpen,
  } = useModelTraceExplorerContext();
  const { canUseAssistant, isPanelOpen, openPanel, sendMessageWhenReady } = useAssistant();
  // The drawer must go non-modal while the assistant panel is open so its focus
  // trap doesn't yank focus back from the chat composer (e.g. right after
  // "Analyze trace" prefills a prompt). Hosts that inject panel state via context
  // (isAssistantPanelOpen) win; otherwise fall back to the live assistant state.
  const assistantPanelOpen = isAssistantPanelOpen ?? isPanelOpen;

  const enableCustomTraceView = shouldEnableModelTraceExplorerCustomTraceView();
  const isCustomViewEnabled = enableCustomTraceView && Boolean(openCustomViewAssistant);
  const [traceExplorerDisplayMode, setTraceExplorerDisplayMode] = useState<ModelTraceExplorerDisplayMode>('default');
  const canCreateCustomView =
    isCustomViewEnabled && Boolean(customViewDefinition?.canPersist) && !customViewDefinition?.hasReachedViewLimit;
  const handleCreateCustomView = useCallback(() => {
    if (!canCreateCustomView) {
      return;
    }
    customViewDefinition?.startNewView('');
    setTraceExplorerDisplayMode('custom');
  }, [canCreateCustomView, customViewDefinition]);
  const snapDrawerLeft = enableCustomTraceView && Boolean(isAssistantPanelOpen) && !isFullscreen;
  // Non-modal whenever the drawer snaps beside a host-repositioned assistant, OR
  // whenever the live assistant panel is open (OSS: AssistantAwareDrawer flips the
  // drawer to the left itself). Either way the modal focus trap must be off.
  const isDrawerModal = !snapDrawerLeft && !assistantPanelOpen;
  const notificationClassName = getPrefixedClassName('notification');
  const baseDrawerWidth = snapDrawerLeft ? CUSTOM_VIEW_INITIAL_WIDTH : drawerWidth;
  const [resizedWidth, setResizedWidth] = useState<number | string>(baseDrawerWidth);
  const [viewportWidth, setViewportWidth] = useState(() => window.innerWidth);
  const width = isFullscreen ? '100vw' : resizedWidth;
  const drawerWidthPixels = isFullscreen ? viewportWidth : resolveWidthToPixels(resizedWidth, viewportWidth);
  const showTraceMetadata = isFullscreen || drawerWidthPixels >= TRACE_METADATA_MIN_WIDTH;
  const compactPrimaryActions = !isFullscreen && drawerWidthPixels < LABELED_PRIMARY_ACTIONS_MIN_WIDTH;
  const resizeHandleOffset = useMemo(
    () => resolveWidthToPixels(resizedWidth, viewportWidth),
    [resizedWidth, viewportWidth],
  );
  const isResizingRef = useRef(false);

  const location = useLocation();
  const locationPath = `${location.pathname}${location.search}${location.hash}`;
  const shareUrl = `${window.location.origin}${locationPath}`;
  const {
    copy: copyShareLink,
    tooltipMessage,
    tooltipOpen,
    handleTooltipOpenChange,
  } = useCopyController(
    shareUrl,
    intl.formatMessage({
      defaultMessage: 'Copy link to trace',
      description: 'Tooltip for the copy link to trace button',
    }),
  );
  const {
    copy: copyTraceId,
    tooltipMessage: traceIdTooltipMessage,
    tooltipOpen: traceIdTooltipOpen,
    handleTooltipOpenChange: handleTraceIdTooltipOpenChange,
  } = useCopyController(
    traceInfo?.trace_id ?? '',
    intl.formatMessage({
      defaultMessage: 'Copy trace ID',
      description: 'Tooltip for copying the trace ID from the trace drawer header',
    }),
  );
  const handleResizePointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    isResizingRef.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  useEffect(() => {
    setResizedWidth(baseDrawerWidth);
  }, [baseDrawerWidth]);

  useEffect(() => {
    const handleViewportResize = () => setViewportWidth(window.innerWidth);
    window.addEventListener('resize', handleViewportResize);
    return () => {
      window.removeEventListener('resize', handleViewportResize);
    };
  }, []);

  const handleResizePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!isResizingRef.current) {
        return;
      }
      const maxWidth = window.innerWidth * MAX_DRAWER_WIDTH_RATIO;
      const nextWidth = snapDrawerLeft ? e.clientX : window.innerWidth - e.clientX;
      setResizedWidth(Math.min(Math.max(nextWidth, MIN_DRAWER_WIDTH), maxWidth));
    },
    [snapDrawerLeft],
  );

  const stopResizing = useCallback(() => {
    isResizingRef.current = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  const handleResizePointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
      stopResizing();
    },
    [stopResizing],
  );

  useEffect(
    () => () => {
      if (isResizingRef.current) {
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
    },
    [],
  );

  const finishClose = useCallback(() => {
    if (!isClosingRef.current) {
      return;
    }
    isClosingRef.current = false;
    if (closeTimerRef.current !== undefined) {
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = undefined;
    }
    handleClose();
  }, [handleClose]);

  const beginClose = useCallback(() => {
    if (isClosingRef.current) {
      return;
    }
    isClosingRef.current = true;
    setIsDrawerOpen(false);
    closeTimerRef.current = window.setTimeout(finishClose, DRAWER_CLOSE_ANIMATION_MS);
  }, [finishClose]);

  useEffect(
    () => () => {
      if (closeTimerRef.current !== undefined) {
        window.clearTimeout(closeTimerRef.current);
      }
    },
    [],
  );

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement) {
        if (e.target.role === 'tab') {
          return;
        }
        const tagName = e.target?.tagName?.toLowerCase();
        if (tagName === 'input' || tagName === 'textarea' || e.target.isContentEditable) {
          return;
        }
      }
      if (e.key === 'ArrowLeft' && isPreviousAvailable) {
        e.preventDefault();
        e.stopPropagation();
        selectPreviousEval();
      } else if (e.key === 'ArrowRight' && isNextAvailable) {
        e.preventDefault();
        e.stopPropagation();
        selectNextEval();
      }
    },
    [isPreviousAvailable, isNextAvailable, selectPreviousEval, selectNextEval],
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  const showAddToDatasetButton = Boolean(
    (renderManagedAddToDatasetDropdown || renderExportTracesToDatasetsModal) && experimentId && traceInfo,
  );
  const addToDatasetLabel = intl.formatMessage({
    defaultMessage: 'Add to dataset',
    description: 'Accessible label and tooltip for adding a trace to a dataset',
  });
  const sendToReviewLabel = intl.formatMessage({
    defaultMessage: 'Flag for review',
    description: 'Accessible label and tooltip for flagging a trace for review',
  });
  const reviewLabel = intl.formatMessage({
    defaultMessage: 'Review',
    description: 'Visible label for the action that flags a trace for review',
  });
  const addToLabelingSessionLabel = intl.formatMessage({
    defaultMessage: 'Add to labeling session',
    description: 'Accessible label for adding a trace to a labeling session',
  });
  const shareLabel = intl.formatMessage({
    defaultMessage: 'Share',
    description: 'Accessible label for sharing a trace',
  });
  const [isAddToDatasetDropdownOpen, setIsAddToDatasetDropdownOpen] = useState(false);
  const [isAddToDatasetTooltipOpen, setIsAddToDatasetTooltipOpen] = useState(false);
  const handleAddToDatasetClick = useCallback(() => {
    if (renderManagedAddToDatasetDropdown) {
      setIsAddToDatasetDropdownOpen(true);
      setIsAddToDatasetTooltipOpen(false);
    } else {
      setShowDatasetModal(true);
    }
  }, [renderManagedAddToDatasetDropdown]);
  const addToDatasetTrigger = (
    <Button
      componentId="mlflow.evaluations_review.modal.add_to_dataset"
      aria-label={addToDatasetLabel}
      onClick={renderManagedAddToDatasetDropdown ? undefined : handleAddToDatasetClick}
      icon={<DatabaseIcon css={headerIconCss} />}
      size="small"
      css={fullscreenActionButtonCss}
    >
      {isFullscreen ? (
        <FormattedMessage defaultMessage="Dataset" description="Visible label for adding a trace to a dataset" />
      ) : undefined}
    </Button>
  );
  const addToDatasetButton = showAddToDatasetButton ? (
    <Tooltip
      componentId="mlflow.evaluations_review.modal.add_to_dataset.tooltip"
      content={addToDatasetLabel}
      open={renderManagedAddToDatasetDropdown ? !isAddToDatasetDropdownOpen && isAddToDatasetTooltipOpen : undefined}
      onOpenChange={renderManagedAddToDatasetDropdown ? setIsAddToDatasetTooltipOpen : undefined}
    >
      <span css={{ display: 'flex', alignItems: 'center', lineHeight: 0 }}>
        {renderManagedAddToDatasetDropdown
          ? renderManagedAddToDatasetDropdown({
              children: addToDatasetTrigger,
              open: isAddToDatasetDropdownOpen,
              onOpenChange: (open) => {
                setIsAddToDatasetDropdownOpen(open);
                setIsAddToDatasetTooltipOpen(false);
              },
            })
          : addToDatasetTrigger}
      </span>
    </Tooltip>
  ) : null;
  const flagForReviewButton =
    renderAddToReviewQueueDropdown && experimentId && traceInfo ? (
      <Tooltip
        componentId="mlflow.evaluations_review.modal.flag_for_review.tooltip"
        content={sendToReviewLabel}
        open={!isFlagForReviewDropdownOpen && isFlagForReviewTooltipOpen}
        onOpenChange={setIsFlagForReviewTooltipOpen}
      >
        <span css={{ display: 'flex', alignItems: 'center', lineHeight: 0 }}>
          {React.createElement(renderAddToReviewQueueDropdown, {
            selectedTraceInfos: [traceInfo],
            experimentId,
            onCloseDrawer: beginClose,
            open: isFlagForReviewDropdownOpen,
            onOpenChange: (open) => {
              setIsFlagForReviewDropdownOpen(open);
              setIsFlagForReviewTooltipOpen(false);
            },
            children: (
              <Button
                componentId="mlflow.evaluations_review.modal.flag_for_review"
                aria-label={sendToReviewLabel}
                icon={<CheckboxIcon css={headerIconCss} />}
                size="small"
                css={fullscreenActionButtonCss}
              >
                {isFullscreen ? reviewLabel : undefined}
              </Button>
            ),
          })}
        </span>
      </Tooltip>
    ) : null;
  const handleToggleFullscreen = useCallback(() => setIsFullscreen((value) => !value), []);
  const handleFindClick = useCallback(() => setSearchVisible((visible) => !visible), []);
  const analyzeTraceLabel = intl.formatMessage({
    defaultMessage: 'Analyze trace',
    description: 'Button label for asking the assistant to analyze a trace',
  });
  const analyzeTracePrompt =
    traceInfo?.state === 'ERROR'
      ? intl.formatMessage(
          {
            defaultMessage: 'Debug the error in trace {traceId}.',
            description: 'Prompt sent to the assistant for analyzing a failed trace',
          },
          { traceId: traceInfo.trace_id },
        )
      : traceInfo
        ? intl.formatMessage(
            {
              defaultMessage: 'Analyze trace {traceId}.',
              description: 'Prompt sent to the assistant for analyzing a trace',
            },
            { traceId: traceInfo.trace_id },
          )
        : intl.formatMessage({
            defaultMessage: 'Analyze this trace.',
            description: 'Prompt sent to the assistant for analyzing a trace when its ID is unavailable',
          });
  const handleAnalyzeTrace = useCallback(() => {
    openPanel();
    // Send the analyze prompt immediately rather than only prefilling it, so the
    // assistant starts working on the trace without a second click.
    sendMessageWhenReady(analyzeTracePrompt);
  }, [openPanel, sendMessageWhenReady, analyzeTracePrompt]);
  const findInTraceLabel = intl.formatMessage({
    defaultMessage: 'Find in trace',
    description: 'Accessible label and tooltip for opening the trace search row',
  });
  const closeTracePanelLabel = intl.formatMessage({
    defaultMessage: 'Close trace panel',
    description: 'Accessible label and tooltip for closing the trace drawer',
  });
  const previousTraceLabel = intl.formatMessage({
    defaultMessage: 'Previous trace',
    description: 'Accessible label and tooltip for navigating to the previous trace',
  });
  const nextTraceLabel = intl.formatMessage({
    defaultMessage: 'Next trace',
    description: 'Accessible label and tooltip for navigating to the next trace',
  });
  const fullscreenLabel = isFullscreen
    ? intl.formatMessage({
        defaultMessage: 'Exit full screen',
        description: 'Accessible label and tooltip for collapsing the trace drawer from full screen',
      })
    : intl.formatMessage({
        defaultMessage: 'Full screen',
        description: 'Accessible label and tooltip for expanding the trace drawer to full screen',
      });
  const displayedTraceId = traceInfo?.trace_id.replace(/^tr-/, '').slice(0, 8);
  const formattedExecutionDuration = traceInfo?.execution_duration
    ? formatHeaderDuration(traceInfo.execution_duration)
    : undefined;
  const traceTokenUsage = traceInfo ? getTraceTokenUsage(traceInfo) : undefined;
  const totalTokens = traceTokenUsage?.total_tokens;
  const traceCost = traceInfo ? getTraceCost(traceInfo) : undefined;
  const traceStatus =
    traceInfo?.state === 'OK'
      ? {
          label: intl.formatMessage({
            defaultMessage: 'Success',
            description: 'Successful trace status in the trace drawer header',
          }),
          color: theme.colors.textValidationSuccess,
          textColor: 'success' as const,
          icon: <CheckCircleIcon css={statusIconCss} />,
        }
      : traceInfo?.state === 'ERROR'
        ? {
            label: intl.formatMessage({
              defaultMessage: 'Error',
              description: 'Failed trace status in the trace drawer header',
            }),
            color: theme.colors.textValidationDanger,
            textColor: 'error' as const,
            icon: <XCircleIcon css={statusIconCss} />,
          }
        : traceInfo?.state === 'IN_PROGRESS'
          ? {
              label: intl.formatMessage({
                defaultMessage: 'In progress',
                description: 'Running trace status in the trace drawer header',
              }),
              color: theme.colors.textValidationWarning,
              textColor: 'warning' as const,
              icon: <ClockIcon css={statusIconCss} />,
            }
          : null;
  useEffect(() => {
    if (!isCustomViewEnabled && traceExplorerDisplayMode === 'custom') {
      setTraceExplorerDisplayMode('default');
    }
  }, [isCustomViewEnabled, traceExplorerDisplayMode]);

  return (
    <DrawerComponent.Root
      open={isDrawerOpen}
      modal={isDrawerModal}
      onOpenChange={(open) => {
        if (!open) {
          beginClose();
        } else {
          setIsDrawerOpen(true);
        }
      }}
    >
      {/* The drawer's content-node layout lives on the design system Drawer.Content, which
          the drawer's `css` prop cannot reliably reach through the OSS AssistantAwareDrawer
          path. Scope these rules to the content node via its stable data attribute so it
          matches managed: drop the default 16px content padding-top, clip direct children
          so nested panes own their own scrolling, and give the header a fixed height, its
          own horizontal padding, and a divider from the body below with no trailing margin. */}
      <Global
        styles={{
          [`${DRAWER_CONTENT_SELECTOR}`]: {
            paddingTop: 0,
          },
          [`${DRAWER_CONTENT_SELECTOR} > div`]: {
            overflow: 'hidden',
          },
          [`${DRAWER_CONTENT_SELECTOR} > div:first-of-type`]: {
            boxSizing: 'border-box',
            height: 48,
            minHeight: 48,
            padding: `0 ${theme.spacing.md}px`,
            borderBottom: `1px solid ${theme.colors.border}`,
            marginBottom: 0,
          },
        }}
      />
      {isFullscreen && (
        <Global
          styles={{
            '[data-component-id="mlflow.evaluations_review.modal"]': {
              left: '0 !important',
              right: '0 !important',
              width: '100vw !important',
              minWidth: '100vw !important',
              maxWidth: 'none !important',
              height: '100vh !important',
            },
            '[data-drawer-resize-handle="true"]': {
              display: 'none !important',
            },
          }}
        />
      )}
      {isDrawerModal && (
        <Global
          styles={{
            [`.${notificationClassName}`]: {
              // The modal drawer makes body portals inert, but success notifications
              // contain navigation actions that must remain usable while it stays open.
              pointerEvents: 'auto',
            },
          }}
        />
      )}
      <DrawerComponent.Content
        componentId="mlflow.evaluations_review.modal"
        position={snapDrawerLeft ? 'left' : undefined}
        onInteractOutside={(event) => {
          const target = event.target as HTMLElement;
          if (target?.closest('[data-drawer-resize-handle="true"]') || snapDrawerLeft) {
            event.preventDefault();
          }
        }}
        width={width}
        hideClose
        title={
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              minWidth: 0,
              width: '100%',
              height: '100%',
            }}
          >
            <Tooltip componentId="mlflow.evaluations_review.modal.previous-tooltip" content={previousTraceLabel}>
              <Button
                componentId="mlflow.evaluations_review.modal.previous_eval"
                aria-label={previousTraceLabel}
                icon={<ChevronLeftIcon css={headerIconCss} />}
                disabled={!isPreviousAvailable}
                onClick={() => selectPreviousEval()}
                size="small"
              />
            </Tooltip>
            <Tooltip componentId="mlflow.evaluations_review.modal.next-tooltip" content={nextTraceLabel}>
              <Button
                componentId="mlflow.evaluations_review.modal.next_eval"
                aria-label={nextTraceLabel}
                icon={<ChevronRightIcon css={headerIconCss} />}
                disabled={!isNextAvailable}
                onClick={() => selectNextEval()}
                size="small"
              />
            </Tooltip>
            <Tooltip componentId="mlflow.evaluations_review.modal.fullscreen-tooltip" content={fullscreenLabel}>
              <Button
                componentId="mlflow.evaluations_review.modal.fullscreen"
                aria-label={fullscreenLabel}
                icon={
                  isFullscreen ? <FullscreenExitIcon css={headerIconCss} /> : <FullscreenIcon css={headerIconCss} />
                }
                onClick={handleToggleFullscreen}
                size="small"
              />
            </Tooltip>
            <div
              css={{
                width: 1,
                height: theme.spacing.lg,
                flexShrink: 0,
                backgroundColor: theme.colors.border,
                marginLeft: theme.spacing.xs,
                marginRight: theme.spacing.xs,
              }}
            />
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                minWidth: 0,
                flex: '1 1 auto',
              }}
            >
              <Typography.Text css={{ whiteSpace: 'nowrap' }}>
                <FormattedMessage defaultMessage="Trace" description="Title for the trace details drawer" />
              </Typography.Text>
              {displayedTraceId && traceInfo && (
                <Tooltip
                  componentId="mlflow.evaluations_review.modal.trace-id-tooltip"
                  content={traceIdTooltipMessage}
                  open={traceIdTooltipOpen}
                  onOpenChange={handleTraceIdTooltipOpenChange}
                  maxWidth={400}
                >
                  <Tag
                    componentId="mlflow.evaluations_review.modal.trace-id"
                    color="default"
                    onClick={copyTraceId}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        copyTraceId();
                      }
                    }}
                    role="button"
                    tabIndex={0}
                    css={{ cursor: 'pointer', margin: 0, maxWidth: 96 }}
                  >
                    <Typography.Text
                      size="sm"
                      color="secondary"
                      css={{ fontFamily: 'monospace', whiteSpace: 'nowrap' }}
                    >
                      {displayedTraceId}
                    </Typography.Text>
                  </Tag>
                </Tooltip>
              )}
              {showTraceMetadata && traceStatus && (
                <>
                  <div
                    css={{
                      width: 1,
                      height: theme.spacing.lg,
                      flexShrink: 0,
                      backgroundColor: theme.colors.border,
                      marginLeft: theme.spacing.xs,
                      marginRight: theme.spacing.xs,
                    }}
                  />
                  <div
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: theme.spacing.xs,
                      color: traceStatus.color,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {traceStatus.icon}
                    <Typography.Text size="sm" color={traceStatus.textColor}>
                      {traceStatus.label}
                    </Typography.Text>
                  </div>
                </>
              )}
              {showTraceMetadata && formattedExecutionDuration && (
                <div
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: theme.spacing.xs,
                    color: theme.colors.textSecondary,
                    whiteSpace: 'nowrap',
                  }}
                >
                  <ClockIcon css={statusIconCss} />
                  <Typography.Text size="sm" color="secondary">
                    {formattedExecutionDuration}
                  </Typography.Text>
                </div>
              )}
              {showTraceMetadata && typeof totalTokens === 'number' && Number.isFinite(totalTokens) && (
                <TokenUsageMetadataItem tokenUsage={{ ...traceTokenUsage, total_tokens: totalTokens }} />
              )}
              {showTraceMetadata && isTraceCostType(traceCost) && (
                <CostMetadataItem cost={traceCost} formatTotalCost={formatHeaderCost} />
              )}
            </div>
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md, flexShrink: 0 }}>
              <Tooltip componentId="mlflow.evaluations_review.modal.find-tooltip" content={findInTraceLabel}>
                <Button
                  componentId="mlflow.evaluations_review.modal.find-button"
                  icon={<SearchIcon css={headerIconCss} />}
                  onClick={handleFindClick}
                  aria-pressed={isSearchVisible}
                  aria-label={findInTraceLabel}
                  size="small"
                  css={fullscreenActionButtonCss}
                >
                  {isFullscreen ? (
                    <FormattedMessage
                      defaultMessage="Find"
                      description="Visible label for finding content in a trace"
                    />
                  ) : undefined}
                </Button>
              </Tooltip>
              {addToDatasetButton}
              {flagForReviewButton}
              <Tooltip
                componentId="mlflow.evaluations_review.modal.copy-link-tooltip"
                content={tooltipMessage}
                open={tooltipOpen}
                onOpenChange={handleTooltipOpenChange}
              >
                <Button
                  componentId="mlflow.evaluations_review.modal.share-button"
                  aria-label={shareLabel}
                  icon={<LinkIcon css={headerIconCss} />}
                  onClick={copyShareLink}
                  size="small"
                  css={fullscreenActionButtonCss}
                >
                  {isFullscreen ? (
                    <FormattedMessage
                      defaultMessage="Share"
                      description="Visible label for copying a link to a trace"
                    />
                  ) : undefined}
                </Button>
              </Tooltip>
            </div>
            <ModelTraceExplorerCustomViewSelector
              value={isCustomViewEnabled ? traceExplorerDisplayMode : 'default'}
              onValueChange={setTraceExplorerDisplayMode}
              onCreateCustomView={handleCreateCustomView}
              isCustomViewEnabled={isCustomViewEnabled}
              canCreateCustomView={canCreateCustomView}
              compact={compactPrimaryActions}
              componentId="mlflow.model_trace_explorer.drawer.custom_view_selector"
            />
            {canUseAssistant && (
              <ModelTraceExplorerAssistantButton
                componentId="mlflow.evaluations_review.modal.analyze-trace"
                onClick={handleAnalyzeTrace}
                ariaLabel={compactPrimaryActions ? analyzeTraceLabel : undefined}
              >
                {compactPrimaryActions ? undefined : analyzeTraceLabel}
              </ModelTraceExplorerAssistantButton>
            )}
            <div
              css={{
                width: 1,
                height: theme.spacing.lg,
                flexShrink: 0,
                backgroundColor: theme.colors.border,
                marginLeft: theme.spacing.xs,
                marginRight: theme.spacing.xs,
              }}
            />
            <Tooltip componentId="mlflow.evaluations_review.modal.close-tooltip" content={closeTracePanelLabel}>
              <Button
                componentId="mlflow.evaluations_review.modal.close"
                aria-label={closeTracePanelLabel}
                icon={<CloseIcon css={headerIconCss} />}
                onClick={beginClose}
                size="small"
              />
            </Tooltip>
          </div>
        }
        expandContentToFullHeight
        disableOpenAutoFocus
        css={[
          {
            paddingTop: 0,
            ...(isFullscreen
              ? {
                  left: 0,
                  right: 0,
                  bottom: 0,
                  width: '100vw !important',
                  minWidth: '100vw !important',
                  maxWidth: 'none !important',
                  height: '100vh',
                }
              : {}),
            '&>div': {
              overflow: 'hidden',
            },
            '&>div:first-of-type': {
              boxSizing: 'border-box',
              height: 48,
              minHeight: 48,
              padding: `0 ${theme.spacing.md}px`,
              borderBottom: `1px solid ${theme.colors.border}`,
              marginBottom: 0,
              '&>button': {
                flexShrink: 0,
              },
            },
            '@media (prefers-reduced-motion: no-preference)': {
              '&[data-state="closed"]': {
                animation: `${drawerSlideOutAnimation} ${DRAWER_CLOSE_ANIMATION_MS}ms ease-in forwards`,
              },
            },
          },
        ]}
      >
        <ApplyDesignSystemContextOverrides zIndexBase={2 * theme.options.zIndexBase}>
          <ModelTraceExplorerRightPaneHeaderActionsProvider
            openAddToDatasetModal={showAddToDatasetButton ? handleAddToDatasetClick : undefined}
            experimentId={experimentId}
            isSearchVisible={isSearchVisible}
            traceExplorerDisplayMode={isCustomViewEnabled ? traceExplorerDisplayMode : 'default'}
            setTraceExplorerDisplayMode={setTraceExplorerDisplayMode}
          >
            {isLoading ? <ModelTraceExplorerSkeleton /> : <>{children}</>}
          </ModelTraceExplorerRightPaneHeaderActionsProvider>
        </ApplyDesignSystemContextOverrides>
        {renderExportTracesToDatasetsModal?.({
          selectedTraceInfos: traceInfo ? [traceInfo] : [],
          experimentId: experimentId ?? '',
          visible: showDatasetModal,
          setVisible: setShowDatasetModal,
        })}
      </DrawerComponent.Content>
      {!isFullscreen &&
        createPortal(
          <div
            role="separator"
            aria-orientation="vertical"
            aria-label={intl.formatMessage({
              defaultMessage: 'Resize trace drawer',
              description: 'Accessible label for the trace explorer drawer resize handle',
            })}
            data-drawer-resize-handle="true"
            onPointerDown={handleResizePointerDown}
            onPointerMove={handleResizePointerMove}
            onPointerUp={handleResizePointerUp}
            onPointerCancel={handleResizePointerUp}
            css={{
              position: 'fixed',
              top: 0,
              bottom: 0,
              ...(snapDrawerLeft ? { left: resizeHandleOffset - 6 } : { right: resizeHandleOffset - 6 }),
              width: 12,
              cursor: 'col-resize',
              touchAction: 'none',
              // Modal drawers make body inert; this handle is portaled outside
              // the drawer content, so it must opt back into pointer events.
              pointerEvents: 'auto',
              zIndex: theme.options.zIndexBase + 3,
              '&::after': {
                content: '""',
                position: 'absolute',
                top: 0,
                bottom: 0,
                left: '50%',
                width: 2,
                transform: 'translateX(-50%)',
                backgroundColor: 'transparent',
                borderRadius: 1,
                transition: 'background-color 0.15s',
              },
              '&:hover::after': {
                backgroundColor: theme.colors.borderDecorative,
              },
              '&:active::after': {
                backgroundColor: theme.colors.border,
              },
            }}
          />,
          document.body,
        )}
    </DrawerComponent.Root>
  );
};
