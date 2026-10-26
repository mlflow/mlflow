import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import {
  ApplyDesignSystemContextOverrides,
  ArrowsUpDownIcon,
  Button,
  CheckCircleIcon,
  CheckboxIcon,
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CloseIcon,
  ClockIcon,
  DatabaseIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  FullscreenExitIcon,
  FullscreenIcon,
  LinkIcon,
  PlusIcon,
  SearchIcon,
  Tag,
  ToggleButton,
  Tooltip,
  TruncationTooltip,
  Typography,
  useDesignSystemTheme,
  XCircleIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { Global, keyframes } from '@emotion/react';
import { useAssistant } from '@mlflow/mlflow/src/assistant';

import { CREATE_CUSTOM_VIEW_QUERY_PARAM } from '../constants';
import { shouldEnableModelTraceExplorerCustomTraceView } from '../FeatureUtils';
import { shouldEnableNewEvalRunsUi } from '../../../../common/utils/FeatureUtils';
import { formatEstimatedCostUSD } from '../CostUtils';
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
import { formatExecutionDuration } from './timeline-tree/TimelineTree.utils';
import { useCopyController } from '../../copy/useCopyController';
import { useLocation, useNavigate } from '../RoutingUtils';
import { RunColorCircle } from '../../genai-traces-table/components/RunColorCircle';

const CUSTOM_VIEW_INITIAL_WIDTH = '50vw';
const MIN_DRAWER_WIDTH = 480;
const MAX_DRAWER_WIDTH_RATIO = 0.9;
const DRAWER_CLOSE_ANIMATION_MS = 180;
const TRACE_METADATA_MIN_WIDTH = 960;
const LABELED_PRIMARY_ACTIONS_MIN_WIDTH = 800;
// The new eval runs UI keeps action labels down to a narrower drawer.
const NEW_UI_LABELED_PRIMARY_ACTIONS_MIN_WIDTH = 720;
// Below this width the new UI hides the "Trace" title text (kept for screen readers) so the trace ID fits beside the actions.
const NEW_UI_TRACE_TITLE_MIN_WIDTH = 560;
// Targets the design system Drawer.Content node (it renders these data attributes from the
// Content componentId below), used by the Global rules that reproduce managed's header layout.
const DRAWER_CONTENT_SELECTOR =
  '[data-component-type="drawer_content"][data-component-id="mlflow.evaluations_review.modal"]';
const headerIconCss = { svg: { width: 15, height: 15 } };
const statusIconCss = { svg: { width: 12, height: 12 } };

const formatHeaderDuration = (value: string): string => {
  // The new eval runs UI formats the header duration like span durations (e.g. "2.20s", "71.57ms").
  if (shouldEnableNewEvalRunsUi()) {
    return formatExecutionDuration(value);
  }

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

const getTraceIdPreview = (traceId: string): string => {
  const shortTraceId = traceId.split('/').at(-1) ?? traceId;
  return shortTraceId.replace(/^tr-/, '').slice(0, 8);
};

const formatHeaderCost = (cost: number): string =>
  shouldEnableNewEvalRunsUi() ? formatEstimatedCostUSD(cost, 4) : `$${cost.toFixed(2)}`;

// Targets where a plain letter key belongs to the focused control, so the drawer's single-key
// shortcuts must not fire.
const SHORTCUT_INTERACTIVE_TARGET_SELECTOR = [
  'input',
  'select',
  'textarea',
  '[contenteditable="true"]',
  '[role="checkbox"]',
  '[role="combobox"]',
  '[role="menuitem"]',
  '[role="menuitemcheckbox"]',
  '[role="menuitemradio"]',
  '[role="option"]',
  '[role="radio"]',
  '[role="searchbox"]',
  '[role="slider"]',
  '[role="spinbutton"]',
  '[role="switch"]',
  '[role="tab"]',
  '[role="textbox"]',
  '[role="treeitem"]',
  '[data-radix-popper-content-wrapper]',
  '[data-radix-menu-content]',
  '[data-radix-select-content]',
].join(', ');

const isShortcutInteractiveTarget = (target: EventTarget | null): boolean => {
  if (!(target instanceof Element)) {
    return false;
  }
  return target.closest(SHORTCUT_INTERACTIVE_TARGET_SELECTOR) !== null;
};

interface DrawerContainerBounds {
  top: number;
  right: number;
  bottom: number;
  left: number;
  width: number;
  height: number;
}

const getViewportBounds = (): DrawerContainerBounds => ({
  top: 0,
  right: window.innerWidth,
  bottom: window.innerHeight,
  left: 0,
  width: window.innerWidth,
  height: window.innerHeight,
});

// A drawer scoped to a host container (the MLflow panel beside the open Assistant) stays open when
// the user interacts outside that container, e.g. with the Assistant itself.
const shouldKeepScopedDrawerOpenForOutsideInteraction = (target: EventTarget | null, drawerScope?: HTMLElement) => {
  if (!(target instanceof Node)) {
    return false;
  }
  if (drawerScope && !drawerScope.contains(target)) {
    return true;
  }
  return (
    target instanceof Element &&
    Boolean(target.closest('[data-assistant-ui="true"], [data-drawer-resize-handle="true"]'))
  );
};

const resolveWidthToPixels = (width: number | string, viewportWidth = window.innerWidth): number => {
  if (typeof width === 'number') {
    return width;
  }
  if (width.endsWith('vw')) {
    return (parseFloat(width) / 100) * viewportWidth;
  }
  if (width.endsWith('%')) {
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
  /** Replaces the "Trace" title and the trace id tag, e.g. with a session's title. */
  headerTitle?: React.ReactNode;
  renderManagedAddToDatasetDropdown?: (params: {
    children: React.ReactNode;
    onOpenChange: (open: boolean) => void;
    open: boolean;
  }) => React.ReactNode;
  traceComparison?: {
    baselineTraceId: string;
    candidateTraceId: string;
    baselineTraceUrl: string;
    candidateTraceUrl: string;
    onSwap: () => void;
    baselineTraceInfo?: ModelTraceInfoV3;
    candidateTraceInfo?: ModelTraceInfoV3;
    runOptions?: {
      runUuid: string;
      runName: string;
      traceId?: string;
    }[];
    onSelectBaselineTrace?: (traceId: string) => void;
    onSelectCandidateTrace?: (traceId: string) => void;
    isReviewMode?: boolean;
    onReviewModeChange?: (isReviewMode: boolean) => void;
  };
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
  headerTitle,
  renderManagedAddToDatasetDropdown,
  traceComparison,
}: ModelTraceExplorerDrawerProps): JSX.Element => {
  const { getPrefixedClassName, theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { openAssistant: openCustomViewAssistant } = useCustomViewAssistantConnector();
  const customViewDefinition = useOptionalCustomViewDefinition();
  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState(true);
  const [isSearchVisible, setSearchVisible] = useState(false);
  const [isFlagForReviewDropdownOpen, setIsFlagForReviewDropdownOpen] = useState(false);
  const [isFlagForReviewTooltipOpen, setIsFlagForReviewTooltipOpen] = useState(false);
  const closeTimerRef = useRef<number | undefined>(undefined);
  const isClosingRef = useRef(false);
  const {
    renderExportTracesToDatasetsModal,
    renderAddToReviewQueueDropdown,
    DrawerComponent,
    drawerWidth = '90vw',
    isAssistantPanelOpen,
    getDrawerPortalContainer,
    initiallyFullscreen,
  } = useModelTraceExplorerContext();
  const isNewEvalRunsUi = shouldEnableNewEvalRunsUi();
  // With the new eval runs UI, a host can scope the drawer to a container (the MLflow panel beside
  // the open Assistant): it then opens fullscreen within that container instead of the viewport.
  const isDrawerScoped = isNewEvalRunsUi && Boolean(getDrawerPortalContainer);
  const [isFullscreen, setIsFullscreen] = useState(() => isNewEvalRunsUi && Boolean(initiallyFullscreen));
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
    isCustomViewEnabled &&
    Boolean(customViewDefinition?.isLoaded) &&
    Boolean(customViewDefinition?.canPersist) &&
    !customViewDefinition?.hasReachedViewLimit;
  const handleCreateCustomView = useCallback(() => {
    if (!canCreateCustomView) {
      return;
    }
    customViewDefinition?.startNewView('');
    setTraceExplorerDisplayMode('custom');
  }, [canCreateCustomView, customViewDefinition]);
  const location = useLocation();
  const navigate = useNavigate();
  const createCustomViewRequested = new URLSearchParams(location.search).get(CREATE_CUSTOM_VIEW_QUERY_PARAM) === 'true';
  const createCustomViewIntentHandledRef = useRef(false);
  useEffect(() => {
    if (!createCustomViewRequested) {
      createCustomViewIntentHandledRef.current = false;
      return;
    }
    if (!canCreateCustomView || createCustomViewIntentHandledRef.current) {
      return;
    }

    createCustomViewIntentHandledRef.current = true;
    handleCreateCustomView();

    const nextSearchParams = new URLSearchParams(location.search);
    nextSearchParams.delete(CREATE_CUSTOM_VIEW_QUERY_PARAM);
    const nextSearch = nextSearchParams.toString();
    navigate(
      { pathname: location.pathname, search: nextSearch ? `?${nextSearch}` : '', hash: location.hash },
      { replace: true },
    );
  }, [canCreateCustomView, createCustomViewRequested, handleCreateCustomView, location, navigate]);
  const snapDrawerLeft = !isDrawerScoped && enableCustomTraceView && Boolean(isAssistantPanelOpen) && !isFullscreen;
  // Non-modal whenever the drawer snaps beside a host-repositioned assistant, OR
  // whenever the live assistant panel is open (OSS: AssistantAwareDrawer flips the
  // drawer to the left itself). Either way the modal focus trap must be off.
  const isDrawerModal = !snapDrawerLeft && !assistantPanelOpen && !isDrawerScoped;
  const notificationClassName = getPrefixedClassName('notification');
  const baseDrawerWidth = snapDrawerLeft ? CUSTOM_VIEW_INITIAL_WIDTH : drawerWidth;
  const [resizedWidth, setResizedWidth] = useState<number | string>(baseDrawerWidth);
  const [viewportWidth, setViewportWidth] = useState(() => window.innerWidth);
  const [drawerContainerBounds, setDrawerContainerBounds] = useState(getViewportBounds);
  // A scoped drawer measures against its container; otherwise against the viewport.
  const drawerContainerWidth = isDrawerScoped ? drawerContainerBounds.width : viewportWidth;
  const width = isFullscreen ? (isDrawerScoped ? '100%' : '100vw') : resizedWidth;
  const drawerWidthPixels = isFullscreen
    ? drawerContainerWidth
    : resolveWidthToPixels(resizedWidth, drawerContainerWidth);
  // The new eval runs UI sizes the header by the drawer width in fullscreen too, so a narrow window
  // gets compact, icon-only actions and no trace metadata instead of an overcrowded header.
  const hideTraceTitleText = isNewEvalRunsUi && drawerWidthPixels < NEW_UI_TRACE_TITLE_MIN_WIDTH;
  const showTraceMetadata = isNewEvalRunsUi
    ? drawerWidthPixels >= TRACE_METADATA_MIN_WIDTH
    : isFullscreen || drawerWidthPixels >= TRACE_METADATA_MIN_WIDTH;
  const showFullscreenActionLabels = isNewEvalRunsUi
    ? isFullscreen && !isDrawerScoped && drawerWidthPixels >= NEW_UI_LABELED_PRIMARY_ACTIONS_MIN_WIDTH
    : isFullscreen;
  const compactPrimaryActions = isNewEvalRunsUi
    ? isDrawerScoped || drawerWidthPixels < NEW_UI_LABELED_PRIMARY_ACTIONS_MIN_WIDTH
    : !isFullscreen && drawerWidthPixels < LABELED_PRIMARY_ACTIONS_MIN_WIDTH;
  const fullscreenActionButtonCss = showFullscreenActionLabels
    ? {
        '&&, &&:hover, &&:active': {
          borderColor: 'transparent !important',
        },
        '&& > .anticon + span': {
          marginLeft: `${theme.spacing.sm}px !important`,
        },
      }
    : undefined;
  const resizeHandleOffset = useMemo(
    () => resolveWidthToPixels(resizedWidth, drawerContainerWidth),
    [resizedWidth, drawerContainerWidth],
  );
  const isResizingRef = useRef(false);

  // Subscribe to route changes so the link stays current, but read `window.location.href`: under a
  // hash router the router location omits the `#` and any path prefix the app is served from.
  useLocation();
  const shareUrl = window.location.href;
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

  useLayoutEffect(() => {
    if (isNewEvalRunsUi) {
      setIsFullscreen(Boolean(initiallyFullscreen));
    }
  }, [initiallyFullscreen, isNewEvalRunsUi]);

  useLayoutEffect(() => {
    if (traceComparison?.isReviewMode) {
      setIsFullscreen(true);
    }
  }, [traceComparison?.isReviewMode]);

  useLayoutEffect(() => {
    if (!isDrawerScoped) {
      return undefined;
    }
    const drawerContainer = getDrawerPortalContainer?.();
    const updateDrawerContainerBounds = () => {
      const bounds = drawerContainer?.getBoundingClientRect();
      setDrawerContainerBounds(
        bounds
          ? {
              top: bounds.top,
              right: bounds.right,
              bottom: bounds.bottom,
              left: bounds.left,
              width: bounds.width,
              height: bounds.height,
            }
          : getViewportBounds(),
      );
    };
    updateDrawerContainerBounds();
    const resizeObserver =
      drawerContainer && typeof ResizeObserver !== 'undefined'
        ? new ResizeObserver(updateDrawerContainerBounds)
        : undefined;
    if (drawerContainer) {
      resizeObserver?.observe(drawerContainer);
    }
    window.addEventListener('resize', updateDrawerContainerBounds);
    window.addEventListener('scroll', updateDrawerContainerBounds, true);
    return () => {
      resizeObserver?.disconnect();
      window.removeEventListener('resize', updateDrawerContainerBounds);
      window.removeEventListener('scroll', updateDrawerContainerBounds, true);
    };
  }, [getDrawerPortalContainer, isDrawerScoped]);

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
      if (isDrawerScoped) {
        // A scoped drawer resizes within its container, up to the container's full width.
        const drawerContainerRect = getDrawerPortalContainer?.().getBoundingClientRect();
        const containerLeft = drawerContainerRect?.left ?? 0;
        const containerRight = drawerContainerRect?.right ?? window.innerWidth;
        const containerWidth = drawerContainerRect?.width ?? window.innerWidth;
        const nextWidth = snapDrawerLeft ? e.clientX - containerLeft : containerRight - e.clientX;
        setResizedWidth(Math.min(Math.max(nextWidth, MIN_DRAWER_WIDTH), containerWidth));
        return;
      }
      const maxWidth = window.innerWidth * MAX_DRAWER_WIDTH_RATIO;
      const nextWidth = snapDrawerLeft ? e.clientX : window.innerWidth - e.clientX;
      setResizedWidth(Math.min(Math.max(nextWidth, MIN_DRAWER_WIDTH), maxWidth));
    },
    [getDrawerPortalContainer, isDrawerScoped, snapDrawerLeft],
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

  const showAddToDatasetButton = Boolean(
    !traceComparison &&
    (renderManagedAddToDatasetDropdown || renderExportTracesToDatasetsModal) &&
    experimentId &&
    traceInfo,
  );
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
  const canFlagForReview = Boolean(renderAddToReviewQueueDropdown && experimentId && traceInfo);
  const handleReviewModeChange = useCallback(
    (isReviewMode: boolean) => {
      if (isReviewMode) {
        setIsFullscreen(true);
      }
      traceComparison?.onReviewModeChange?.(isReviewMode);
    },
    [traceComparison],
  );

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (isNewEvalRunsUi && (!isDrawerOpen || isClosingRef.current)) {
        return;
      }
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
      } else if (isNewEvalRunsUi && !e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey && !e.repeat) {
        // The new eval runs UI opens the dataset (d) and review (r) actions from the keyboard. In a trace
        // comparison, which has neither action, r toggles pairwise review instead.
        if (
          showDatasetModal ||
          isAddToDatasetDropdownOpen ||
          isFlagForReviewDropdownOpen ||
          isShortcutInteractiveTarget(e.target)
        ) {
          return;
        }
        if (traceComparison) {
          if (e.key === 'r' && traceComparison.onReviewModeChange) {
            e.preventDefault();
            handleReviewModeChange(!traceComparison.isReviewMode);
          }
        } else if (e.key === 'd' && showAddToDatasetButton) {
          e.preventDefault();
          setIsFlagForReviewDropdownOpen(false);
          handleAddToDatasetClick();
        } else if (e.key === 'r' && canFlagForReview) {
          e.preventDefault();
          setIsAddToDatasetDropdownOpen(false);
          setIsFlagForReviewDropdownOpen(true);
          setIsFlagForReviewTooltipOpen(false);
        }
      }
    },
    [
      isPreviousAvailable,
      isNextAvailable,
      selectPreviousEval,
      selectNextEval,
      isNewEvalRunsUi,
      traceComparison,
      isDrawerOpen,
      showDatasetModal,
      isAddToDatasetDropdownOpen,
      isFlagForReviewDropdownOpen,
      showAddToDatasetButton,
      handleAddToDatasetClick,
      canFlagForReview,
      handleReviewModeChange,
    ],
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

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
  const pairwiseReviewLabel = intl.formatMessage({
    defaultMessage: 'Review',
    description: 'Action that opens pairwise review for the compared traces',
  });
  const pairwiseReviewTooltip = intl.formatMessage(
    {
      defaultMessage: '{label} (R)',
      description: 'Tooltip for the pairwise review action with its keyboard shortcut',
    },
    { label: pairwiseReviewLabel },
  );
  const addToLabelingSessionLabel = intl.formatMessage({
    defaultMessage: 'Add to labeling session',
    description: 'Accessible label for adding a trace to a labeling session',
  });
  const shareLabel = intl.formatMessage({
    defaultMessage: 'Share',
    description: 'Accessible label for sharing a trace',
  });
  const addToDatasetTrigger = (
    <Button
      componentId="mlflow.evaluations_review.modal.add_to_dataset"
      aria-label={addToDatasetLabel}
      onClick={renderManagedAddToDatasetDropdown ? undefined : handleAddToDatasetClick}
      icon={<DatabaseIcon css={headerIconCss} />}
      size="small"
      css={fullscreenActionButtonCss}
    >
      {showFullscreenActionLabels ? (
        <FormattedMessage defaultMessage="Dataset" description="Visible label for adding a trace to a dataset" />
      ) : undefined}
    </Button>
  );
  const addToDatasetButton = showAddToDatasetButton ? (
    <Tooltip
      componentId="mlflow.evaluations_review.modal.add_to_dataset.tooltip"
      content={
        isNewEvalRunsUi
          ? intl.formatMessage(
              {
                defaultMessage: '{label} · D',
                description: 'Tooltip for adding a trace to a dataset with the keyboard shortcut',
              },
              { label: addToDatasetLabel },
            )
          : addToDatasetLabel
      }
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
    !traceComparison && renderAddToReviewQueueDropdown && experimentId && traceInfo ? (
      <Tooltip
        componentId="mlflow.evaluations_review.modal.flag_for_review.tooltip"
        content={
          isNewEvalRunsUi
            ? intl.formatMessage(
                {
                  defaultMessage: '{label} · R',
                  description: 'Tooltip for flagging a trace for review with the keyboard shortcut',
                },
                { label: sendToReviewLabel },
              )
            : sendToReviewLabel
        }
        open={!isFlagForReviewDropdownOpen && isFlagForReviewTooltipOpen}
        onOpenChange={setIsFlagForReviewTooltipOpen}
      >
        <span css={{ display: 'flex', alignItems: 'center', lineHeight: 0 }}>
          {React.createElement(renderAddToReviewQueueDropdown, {
            selectedTraceInfos: [traceInfo],
            experimentId,
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
                {showFullscreenActionLabels ? reviewLabel : undefined}
              </Button>
            ),
          })}
        </span>
      </Tooltip>
    ) : null;
  const handleToggleFullscreen = useCallback(() => setIsFullscreen((value) => !value), []);
  const handleFindClick = useCallback(() => setSearchVisible((visible) => !visible), []);
  const analyzeTraceLabel = traceComparison
    ? intl.formatMessage({
        defaultMessage: 'Analyze traces',
        description: 'Button label for asking the assistant to compare two traces',
      })
    : intl.formatMessage({
        defaultMessage: 'Analyze trace',
        description: 'Button label for asking the assistant to analyze a trace',
      });
  const analyzeTracePrompt = traceComparison
    ? intl.formatMessage(
        {
          defaultMessage:
            'Compare baseline trace {baselineTraceId} with candidate trace {candidateTraceId}. Highlight differences in their span structure, inputs, outputs, attributes, duration, token usage, and estimated cost.',
          description: 'Prompt sent to the assistant for comparing two traces',
        },
        {
          baselineTraceId: traceComparison.baselineTraceId,
          candidateTraceId: traceComparison.candidateTraceId,
        },
      )
    : traceInfo?.state === 'ERROR'
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
  const baselineRunOption = traceComparison?.runOptions?.find(
    (option) => option.traceId === traceComparison.baselineTraceId,
  );
  const candidateRunOption = traceComparison?.runOptions?.find(
    (option) => option.traceId === traceComparison.candidateTraceId,
  );
  const showComparisonRunSelectors = Boolean(
    traceComparison?.runOptions?.length &&
    baselineRunOption &&
    candidateRunOption &&
    traceComparison.onSelectBaselineTrace &&
    traceComparison.onSelectCandidateTrace,
  );
  const comparisonTraceContext = traceComparison ? (
    <div
      css={{
        display: 'grid',
        gridTemplateColumns: showComparisonRunSelectors
          ? 'max-content minmax(0, 240px) max-content minmax(0, 240px) max-content'
          : 'repeat(5, max-content)',
        alignItems: 'center',
        alignSelf: 'stretch',
        columnGap: theme.spacing.sm,
        flex: '1 1 0',
        width: '100%',
        minWidth: 0,
        overflow: 'hidden',
      }}
    >
      <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs, whiteSpace: 'nowrap' }}>
        <Typography.Text size="sm" color="secondary">
          {showComparisonRunSelectors ? (
            <FormattedMessage
              defaultMessage="Comparing traces from Runs"
              description="Context shown in an evaluation run trace comparison header"
            />
          ) : (
            <FormattedMessage
              defaultMessage="Comparing traces"
              description="Context shown in a trace comparison header"
            />
          )}
        </Typography.Text>
        {traceComparison.baselineTraceId === traceComparison.candidateTraceId && (
          <Tooltip
            componentId="mlflow.trace-explorer.comparison.same-trace-tooltip"
            content={intl.formatMessage({
              defaultMessage: 'Both runs reference the same trace.',
              description: 'Tooltip explaining the same trace badge in a trace comparison header',
            })}
          >
            <Tag componentId="mlflow.trace-explorer.comparison.same-trace-tag">
              <FormattedMessage
                defaultMessage="Same trace"
                description="Badge shown when both sides of a trace comparison reference the same trace"
              />
            </Tag>
          </Tooltip>
        )}
      </span>
      {showComparisonRunSelectors ? (
        <span
          css={{
            display: 'flex',
            alignItems: 'center',
            alignSelf: 'center',
            width: '100%',
            minWidth: 0,
            overflow: 'hidden',
          }}
        >
          <DialogCombobox
            componentId="mlflow.trace-explorer.comparison.baseline-run"
            id="trace-comparison-baseline-run"
            value={[baselineRunOption?.runUuid ?? '']}
          >
            <DialogComboboxCustomButtonTriggerWrapper>
              <Button
                componentId="mlflow.trace-explorer.comparison.baseline-run-trigger"
                type="tertiary"
                size="small"
                block
                dangerouslyAppendWrapperCss={{ width: '100%', minWidth: 0, overflow: 'hidden' }}
                css={{
                  minWidth: 0,
                  maxWidth: '100%',
                  width: '100%',
                  overflow: 'hidden',
                }}
              >
                <span
                  css={{
                    display: 'grid',
                    gridTemplateColumns: `${theme.spacing.sm}px minmax(0, 1fr) auto`,
                    alignItems: 'center',
                    gap: theme.spacing.xs,
                    width: '100%',
                    minWidth: 0,
                    maxWidth: '100%',
                    overflow: 'hidden',
                  }}
                >
                  <span
                    css={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: theme.spacing.sm,
                      height: theme.spacing.sm,
                      flexShrink: 0,
                      '& > label': { transform: 'scale(0.67)', transformOrigin: 'center' },
                    }}
                  >
                    <RunColorCircle color={theme.colors.textSecondary} />
                  </span>
                  <span css={{ display: 'block', width: '100%', minWidth: 0, overflow: 'hidden' }}>
                    <TruncationTooltip componentId="mlflow.trace-explorer.comparison.baseline-run-name-tooltip">
                      <Typography.Text
                        css={{
                          display: 'block',
                          width: '100%',
                          minWidth: 0,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {baselineRunOption?.runName}
                      </Typography.Text>
                    </TruncationTooltip>
                  </span>
                  <ChevronDownIcon css={{ color: theme.colors.textSecondary }} />
                </span>
              </Button>
            </DialogComboboxCustomButtonTriggerWrapper>
            <DialogComboboxContent>
              <DialogComboboxOptionList>
                {traceComparison.runOptions?.map((option) => (
                  <DialogComboboxOptionListSelectItem
                    key={option.runUuid}
                    value={option.runName}
                    checked={option.traceId === traceComparison.baselineTraceId}
                    disabled={!option.traceId || option.traceId === traceComparison.candidateTraceId}
                    onChange={() => option.traceId && traceComparison.onSelectBaselineTrace?.(option.traceId)}
                  >
                    {option.runName}
                  </DialogComboboxOptionListSelectItem>
                ))}
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
        </span>
      ) : (
        <Tooltip
          componentId="mlflow.trace-explorer.comparison.baseline-id-tooltip"
          content={traceComparison.baselineTraceId}
        >
          <Typography.Text size="sm" color="secondary" css={{ whiteSpace: 'nowrap' }}>
            <span
              css={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
                fontFamily: 'monospace',
                whiteSpace: 'nowrap',
              }}
            >
              <span
                css={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: theme.spacing.sm,
                  height: theme.spacing.sm,
                  '& > label': { transform: 'scale(0.67)', transformOrigin: 'center' },
                }}
              >
                <RunColorCircle color={theme.colors.textSecondary} />
              </span>
              <FormattedMessage
                defaultMessage="{traceId}"
                description="Baseline trace identifier in the trace comparison header"
                values={{ traceId: getTraceIdPreview(traceComparison.baselineTraceId) }}
              />
            </span>
          </Typography.Text>
        </Tooltip>
      )}
      <span
        css={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        <Tooltip
          componentId="mlflow.trace-explorer.comparison.swap-tooltip"
          content={intl.formatMessage({
            defaultMessage: 'Swap baseline and candidate',
            description: 'Tooltip for reversing the trace comparison order',
          })}
        >
          <Button
            componentId="mlflow.trace-explorer.comparison.swap"
            size="small"
            icon={<ArrowsUpDownIcon rotate={90} css={headerIconCss} />}
            aria-label={intl.formatMessage({
              defaultMessage: 'Swap baseline and candidate',
              description: 'Accessible label for reversing the trace comparison order',
            })}
            onClick={traceComparison.onSwap}
          />
        </Tooltip>
      </span>
      {showComparisonRunSelectors ? (
        <span
          css={{
            display: 'flex',
            alignItems: 'center',
            alignSelf: 'center',
            width: '100%',
            minWidth: 0,
            overflow: 'hidden',
          }}
        >
          <DialogCombobox
            componentId="mlflow.trace-explorer.comparison.candidate-run"
            id="trace-comparison-candidate-run"
            value={[candidateRunOption?.runUuid ?? '']}
          >
            <DialogComboboxCustomButtonTriggerWrapper>
              <Button
                componentId="mlflow.trace-explorer.comparison.candidate-run-trigger"
                type="tertiary"
                size="small"
                block
                dangerouslyAppendWrapperCss={{ width: '100%', minWidth: 0, overflow: 'hidden' }}
                css={{
                  minWidth: 0,
                  maxWidth: '100%',
                  width: '100%',
                  overflow: 'hidden',
                }}
              >
                <span
                  css={{
                    display: 'grid',
                    gridTemplateColumns: `${theme.spacing.sm}px minmax(0, 1fr) auto`,
                    alignItems: 'center',
                    gap: theme.spacing.xs,
                    width: '100%',
                    minWidth: 0,
                    maxWidth: '100%',
                    overflow: 'hidden',
                  }}
                >
                  <span
                    css={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: theme.spacing.sm,
                      height: theme.spacing.sm,
                      flexShrink: 0,
                      '& > label': { transform: 'scale(0.67)', transformOrigin: 'center' },
                    }}
                  >
                    <RunColorCircle color={theme.colors.actionLinkDefault} />
                  </span>
                  <span css={{ display: 'block', width: '100%', minWidth: 0, overflow: 'hidden' }}>
                    <TruncationTooltip componentId="mlflow.trace-explorer.comparison.candidate-run-name-tooltip">
                      <Typography.Text
                        css={{
                          display: 'block',
                          width: '100%',
                          minWidth: 0,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {candidateRunOption?.runName}
                      </Typography.Text>
                    </TruncationTooltip>
                  </span>
                  <ChevronDownIcon css={{ color: theme.colors.textSecondary }} />
                </span>
              </Button>
            </DialogComboboxCustomButtonTriggerWrapper>
            <DialogComboboxContent>
              <DialogComboboxOptionList>
                {traceComparison.runOptions?.map((option) => (
                  <DialogComboboxOptionListSelectItem
                    key={option.runUuid}
                    value={option.runName}
                    checked={option.traceId === traceComparison.candidateTraceId}
                    disabled={!option.traceId || option.traceId === traceComparison.baselineTraceId}
                    onChange={() => option.traceId && traceComparison.onSelectCandidateTrace?.(option.traceId)}
                  >
                    {option.runName}
                  </DialogComboboxOptionListSelectItem>
                ))}
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
        </span>
      ) : (
        <Tooltip
          componentId="mlflow.trace-explorer.comparison.candidate-id-tooltip"
          content={traceComparison.candidateTraceId}
        >
          <Typography.Text size="sm" color="secondary" css={{ whiteSpace: 'nowrap' }}>
            <span
              css={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
                fontFamily: 'monospace',
                whiteSpace: 'nowrap',
              }}
            >
              <span
                css={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: theme.spacing.sm,
                  height: theme.spacing.sm,
                  '& > label': { transform: 'scale(0.67)', transformOrigin: 'center' },
                }}
              >
                <RunColorCircle color={theme.colors.actionLinkDefault} />
              </span>
              <FormattedMessage
                defaultMessage="{traceId}"
                description="Candidate trace identifier in the trace comparison header"
                values={{ traceId: getTraceIdPreview(traceComparison.candidateTraceId) }}
              />
            </span>
          </Typography.Text>
        </Tooltip>
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
    </div>
  ) : null;
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
      {isFullscreen && !isDrawerScoped && (
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
        {...(isDrawerScoped ? { isFullscreen, 'data-trace-drawer-fullscreen': isFullscreen } : {})}
        position={snapDrawerLeft ? 'left' : undefined}
        onInteractOutside={(event) => {
          if (isDrawerScoped) {
            if (shouldKeepScopedDrawerOpenForOutsideInteraction(event.target, getDrawerPortalContainer?.())) {
              event.preventDefault();
            }
            return;
          }
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
              {headerTitle ?? (
                <>
                  {!traceComparison && (
                    <Typography.Text
                      css={
                        hideTraceTitleText
                          ? {
                              position: 'absolute',
                              width: 1,
                              height: 1,
                              overflow: 'hidden',
                              clip: 'rect(0 0 0 0)',
                              whiteSpace: 'nowrap',
                            }
                          : { whiteSpace: 'nowrap' }
                      }
                    >
                      <FormattedMessage defaultMessage="Trace" description="Title for the trace details drawer" />
                    </Typography.Text>
                  )}
                  {displayedTraceId && traceInfo && !traceComparison && (
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
                        css={
                          isNewEvalRunsUi
                            ? { cursor: 'pointer', margin: 0, maxWidth: 96, minWidth: 0, overflow: 'hidden' }
                            : { cursor: 'pointer', margin: 0, maxWidth: 96 }
                        }
                      >
                        <Typography.Text
                          size="sm"
                          color="secondary"
                          css={
                            isNewEvalRunsUi
                              ? {
                                  fontFamily: 'monospace',
                                  whiteSpace: 'nowrap',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                }
                              : { fontFamily: 'monospace', whiteSpace: 'nowrap' }
                          }
                        >
                          {displayedTraceId}
                        </Typography.Text>
                      </Tag>
                    </Tooltip>
                  )}
                </>
              )}
              {showTraceMetadata && comparisonTraceContext}
              {showTraceMetadata && !traceComparison && traceStatus && (
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
              {showTraceMetadata && !traceComparison && formattedExecutionDuration && (
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
              {showTraceMetadata &&
                !traceComparison &&
                typeof totalTokens === 'number' &&
                Number.isFinite(totalTokens) && (
                  <TokenUsageMetadataItem tokenUsage={{ ...traceTokenUsage, total_tokens: totalTokens }} />
                )}
              {showTraceMetadata && !traceComparison && isTraceCostType(traceCost) && (
                <CostMetadataItem cost={traceCost} formatTotalCost={formatHeaderCost} />
              )}
            </div>
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: hideTraceTitleText ? theme.spacing.sm : theme.spacing.md,
                flexShrink: 0,
              }}
            >
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
                  {showFullscreenActionLabels ? (
                    <FormattedMessage
                      defaultMessage="Find"
                      description="Visible label for finding content in a trace"
                    />
                  ) : undefined}
                </Button>
              </Tooltip>
              {addToDatasetButton}
              {flagForReviewButton}
              {traceComparison?.onReviewModeChange && (
                <Tooltip componentId="mlflow.trace-comparison.review-tooltip" content={pairwiseReviewTooltip}>
                  <ToggleButton
                    componentId="mlflow.trace-comparison.review"
                    aria-label={pairwiseReviewTooltip}
                    pressed={traceComparison.isReviewMode}
                    onPressedChange={handleReviewModeChange}
                    icon={<CheckboxIcon css={headerIconCss} />}
                    size="small"
                    css={[
                      fullscreenActionButtonCss,
                      {
                        '&&[data-state="on"], &&[data-state="on"]:hover, &&[data-state="on"]:active': {
                          backgroundColor: theme.colors.actionDefaultBackgroundHover,
                        },
                      },
                    ]}
                  >
                    {showFullscreenActionLabels ? pairwiseReviewLabel : undefined}
                  </ToggleButton>
                </Tooltip>
              )}
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
                  {showFullscreenActionLabels ? (
                    <FormattedMessage
                      defaultMessage="Share"
                      description="Visible label for copying a link to a trace"
                    />
                  ) : undefined}
                </Button>
              </Tooltip>
            </div>
            {!traceComparison && (
              <ModelTraceExplorerCustomViewSelector
                value={isCustomViewEnabled ? traceExplorerDisplayMode : 'default'}
                onValueChange={setTraceExplorerDisplayMode}
                onCreateCustomView={handleCreateCustomView}
                isCustomViewEnabled={isCustomViewEnabled}
                canCreateCustomView={canCreateCustomView}
                compact={compactPrimaryActions}
                componentId="mlflow.model_trace_explorer.drawer.custom_view_selector"
              />
            )}
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
            ...(isFullscreen && !isDrawerScoped
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
              ...(isDrawerScoped
                ? {
                    top: drawerContainerBounds.top,
                    height: drawerContainerBounds.height,
                    ...(snapDrawerLeft
                      ? { left: drawerContainerBounds.left + resizeHandleOffset - 6 }
                      : { right: window.innerWidth - drawerContainerBounds.right + resizeHandleOffset - 6 }),
                  }
                : {
                    top: 0,
                    bottom: 0,
                    ...(snapDrawerLeft ? { left: resizeHandleOffset - 6 } : { right: resizeHandleOffset - 6 }),
                  }),
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
          (isDrawerScoped ? getDrawerPortalContainer?.() : undefined) ?? document.body,
        )}
    </DrawerComponent.Root>
  );
};
