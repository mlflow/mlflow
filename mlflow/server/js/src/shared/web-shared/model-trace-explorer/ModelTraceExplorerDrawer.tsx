import React, { useCallback, useEffect, useState } from 'react';

import {
  ApplyDesignSystemContextOverrides,
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  FlagPointerIcon,
  PlusIcon,
  Notification,
  SparkleIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { Global } from '@emotion/react';
import { useAssistant } from '@mlflow/mlflow/src/assistant';

import { ModelTraceExplorerSkeleton } from './ModelTraceExplorerSkeleton';
import { useModelTraceExplorerContext } from './ModelTraceExplorerContext';
import type { ModelTraceInfoV3 } from './ModelTrace.types';
import { getAiGradientBorderStyle } from '../design-system/aiGradientBorderStyle';
import { copyToClipboard } from '../../../common/utils/copyToClipboard';
import { useLocalStorage } from '../hooks/useLocalStorage';

const FLAG_FOR_REVIEW_GUIDANCE_STORAGE_KEY = 'mlflow.flagForReview.guidanceShown';

// Targets the Radix popper wrapper that contains the tooltip background, arrow,
// and content so the entire tooltip fades in together. If Radix renames this
// internal attribute the animation silently stops — no functional breakage.
const RADIX_POPPER_WRAPPER_SELECTOR = '[data-radix-popper-content-wrapper]:has([data-flag-guidance])';
const FLAG_FOR_REVIEW_GUIDANCE_STORAGE_VERSION = 1;

export interface ModelTraceExplorerDrawerProps {
  children: React.ReactNode;
  selectPreviousEval: () => void;
  selectNextEval: () => void;
  isPreviousAvailable: boolean;
  isNextAvailable: boolean;
  renderModalTitle: () => React.ReactNode;
  handleClose: () => void;
  isLoading?: boolean;
  experimentId?: string;
  traceInfo?: ModelTraceInfoV3;
}

export const ModelTraceExplorerDrawer = ({
  selectPreviousEval,
  selectNextEval,
  isPreviousAvailable,
  isNextAvailable,
  renderModalTitle,
  handleClose,
  children,
  isLoading,
  experimentId,
  traceInfo,
}: ModelTraceExplorerDrawerProps) => {
  const { theme } = useDesignSystemTheme();
  const { isLocalServer, openPanel } = useAssistant();
  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [showCopiedNotification, setShowCopiedNotification] = useState(false);
  const [showCopyError, setShowCopyError] = useState(false);
  const {
    renderExportTracesToDatasetsModal,
    renderAddToReviewQueueDropdown,
    DrawerComponent,
    drawerWidth = '60vw',
  } = useModelTraceExplorerContext();

  const [hasSeenFlagGuidance, setHasSeenFlagGuidance] = useLocalStorage({
    key: FLAG_FOR_REVIEW_GUIDANCE_STORAGE_KEY,
    version: FLAG_FOR_REVIEW_GUIDANCE_STORAGE_VERSION,
    initialValue: false,
  });

  const [isDrawerAnimationDone, setIsDrawerAnimationDone] = useState(false);

  const showFlagForReviewButton = Boolean(renderAddToReviewQueueDropdown && experimentId && traceInfo);

  useEffect(() => {
    if (!showFlagForReviewButton || hasSeenFlagGuidance) {
      return;
    }
    const timer = setTimeout(() => setIsDrawerAnimationDone(true), 500);
    return () => clearTimeout(timer);
  }, [showFlagForReviewButton, hasSeenFlagGuidance]);

  const handleDismissFlagGuidance = useCallback(() => {
    setHasSeenFlagGuidance(true);
  }, [setHasSeenFlagGuidance]);

  const handleShareClick = useCallback(async () => {
    const success = await copyToClipboard(window.location.href);
    if (success) {
      setShowCopiedNotification(true);
      setTimeout(() => setShowCopiedNotification(false), 2000);
    } else {
      setShowCopyError(true);
      setTimeout(() => setShowCopyError(false), 2000);
    }
  }, []);

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
        e.stopPropagation();
        selectPreviousEval();
      } else if (e.key === 'ArrowRight' && isNextAvailable) {
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

  const showAddToDatasetButton = Boolean(renderExportTracesToDatasetsModal && experimentId && traceInfo);
  const handleAddToDatasetClick = useCallback(() => setShowDatasetModal(true), []);

  const showFlagGuidance = showFlagForReviewButton && !hasSeenFlagGuidance && isDrawerAnimationDone;

  const flagForReviewButton =
    showFlagForReviewButton && renderAddToReviewQueueDropdown
      ? React.createElement(renderAddToReviewQueueDropdown, {
          selectedTraceInfos: traceInfo ? [traceInfo] : [],
          experimentId: experimentId ?? '',
          onOpenChange: (open: boolean) => {
            if (open && !hasSeenFlagGuidance) {
              handleDismissFlagGuidance();
            }
          },
          children: (
            <Button componentId="mlflow.evaluations_review.modal.flag_for_review" icon={<FlagPointerIcon />}>
              <FormattedMessage
                defaultMessage="Flag for review"
                description="Button text for assigning a trace to reviewers via a review queue"
              />
            </Button>
          ),
        })
      : null;

  return (
    <DrawerComponent.Root
      open
      onOpenChange={(open) => {
        if (!open) {
          handleClose();
        }
      }}
    >
      <DrawerComponent.Content
        componentId="mlflow.evaluations_review.modal"
        width={drawerWidth}
        title={
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
            <Button
              componentId="mlflow.evaluations_review.modal.previous_eval"
              disabled={!isPreviousAvailable}
              onClick={() => selectPreviousEval()}
            >
              <ChevronLeftIcon />
            </Button>
            <Button
              componentId="mlflow.evaluations_review.modal.next_eval"
              disabled={!isNextAvailable}
              onClick={() => selectNextEval()}
            >
              <ChevronRightIcon />
            </Button>
            <div css={{ flex: 1, overflow: 'hidden' }}>{renderModalTitle()}</div>
            {isLocalServer && (
              // data-assistant-ui marks this as assistant UI so AssistantAwareDrawer won't treat
              // the click as an outside-click and close. See AssistantAwareDrawer.tsx.
              <Button
                componentId="mlflow.assistant.trace_header_button"
                data-assistant-ui="true"
                icon={<SparkleIcon color="ai" />}
                onClick={openPanel}
                css={{ flexShrink: 0, ...getAiGradientBorderStyle(theme) }}
              >
                <FormattedMessage
                  defaultMessage="Analyze with Assistant"
                  description="Button that opens the MLflow assistant side panel to analyze the current trace"
                />
              </Button>
            )}
            {showAddToDatasetButton && (
              <Button
                componentId="mlflow.evaluations_review.modal.add_to_dataset"
                onClick={handleAddToDatasetClick}
                icon={<PlusIcon />}
              >
                <FormattedMessage
                  defaultMessage="Add to dataset"
                  description="Button text for adding a trace to a dataset"
                />
              </Button>
            )}
            {showFlagForReviewButton && (
              <>
                {showFlagGuidance && (
                  <Global
                    styles={{
                      '@keyframes flagGuidanceFadeIn': {
                        from: { opacity: 0 },
                        to: { opacity: 1 },
                      },
                      [RADIX_POPPER_WRAPPER_SELECTOR]: {
                        animation: 'flagGuidanceFadeIn 300ms ease-in',
                      },
                    }}
                  />
                )}
                <Tooltip
                  componentId="mlflow.evaluations_review.modal.flag_for_review.guidance"
                  open={showFlagGuidance}
                  content={
                    <div data-flag-guidance onClick={handleDismissFlagGuidance} css={{ cursor: 'pointer' }}>
                      <FormattedMessage
                        defaultMessage="New! Flag traces for review and add them to a review queue."
                        description="Guidance tooltip message for the flag for review button in the trace drawer"
                      />
                    </div>
                  }
                >
                  <div>{flagForReviewButton}</div>
                </Tooltip>
              </>
            )}
            <Tooltip
              componentId="mlflow.evaluations_review.modal.share-tooltip"
              content={
                <FormattedMessage
                  defaultMessage="Copy link to trace"
                  description="Tooltip for the share trace button"
                />
              }
            >
              <Button componentId="mlflow.evaluations_review.modal.share-button" onClick={handleShareClick}>
                <FormattedMessage defaultMessage="Share" description="Label for the share trace button" />
              </Button>
            </Tooltip>
          </div>
        }
        expandContentToFullHeight
        css={[
          {
            '&>div': {
              overflow: 'hidden',
            },
            '&>div:first-child': {
              paddingLeft: theme.spacing.md,
              paddingTop: 1,
              paddingBottom: 1,
              '&>button': {
                flexShrink: 0,
              },
            },
          },
        ]}
      >
        <ApplyDesignSystemContextOverrides zIndexBase={2 * theme.options.zIndexBase}>
          {isLoading ? <ModelTraceExplorerSkeleton /> : <>{children}</>}
        </ApplyDesignSystemContextOverrides>
        {renderExportTracesToDatasetsModal?.({
          selectedTraceInfos: traceInfo ? [traceInfo] : [],
          experimentId: experimentId ?? '',
          visible: showDatasetModal,
          setVisible: setShowDatasetModal,
        })}
      </DrawerComponent.Content>
      {showCopiedNotification && (
        <Notification.Provider>
          <Notification.Root severity="success" componentId="mlflow.evaluations_review.modal.share-notification">
            <Notification.Title>
              <FormattedMessage
                defaultMessage="Copied to clipboard"
                description="Success message after copying trace link"
              />
            </Notification.Title>
          </Notification.Root>
          <Notification.Viewport />
        </Notification.Provider>
      )}
      {showCopyError && (
        <Notification.Provider>
          <Notification.Root severity="error" componentId="mlflow.evaluations_review.modal.share-error-notification">
            <Notification.Title>
              <FormattedMessage
                defaultMessage="Failed to copy to clipboard"
                description="Error message when clipboard copy fails"
              />
            </Notification.Title>
          </Notification.Root>
          <Notification.Viewport />
        </Notification.Provider>
      )}
    </DrawerComponent.Root>
  );
};
