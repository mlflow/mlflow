import React, { useCallback, useEffect, useState } from 'react';

import {
  ApplyDesignSystemContextOverrides,
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  PlusIcon,
  Notification,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerSkeleton } from './ModelTraceExplorerSkeleton';
import { useModelTraceExplorerContext } from './ModelTraceExplorerContext';
import type { ModelTraceInfoV3 } from './ModelTrace.types';

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
  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [showCopiedNotification, setShowCopiedNotification] = useState(false);
  const { renderExportTracesToDatasetsModal, DrawerComponent, drawerWidth = '60vw' } = useModelTraceExplorerContext();

  const handleShareClick = useCallback(() => {
    navigator.clipboard.writeText(window.location.href);
    setShowCopiedNotification(true);
    setTimeout(() => setShowCopiedNotification(false), 2000);
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
    </DrawerComponent.Root>
  );
};
