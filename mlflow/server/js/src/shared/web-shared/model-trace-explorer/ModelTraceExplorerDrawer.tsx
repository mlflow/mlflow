import React, { useCallback, useEffect, useState } from 'react';

import {
  ApplyDesignSystemContextOverrides,
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  PlusIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { AssistantAwareDrawer } from '@mlflow/mlflow/src/common/components/AssistantAwareDrawer';

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
  const { renderExportTracesToDatasetsModal } = useModelTraceExplorerContext();

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

  return (
    <AssistantAwareDrawer.Root
      open
      onOpenChange={(open) => {
        if (!open) {
          handleClose();
        }
      }}
    >
      <AssistantAwareDrawer.Content
        componentId="mlflow.evaluations_review.modal"
        width="90vw"
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
                componentId="mlflow.evaluations_review.modal.add_to_evaluation_dataset"
                onClick={() => setShowDatasetModal(true)}
                icon={<PlusIcon />}
              >
                <FormattedMessage
                  defaultMessage="Add to dataset"
                  description="Button text for adding a trace to a evaluation dataset"
                />
              </Button>
            )}
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
      </AssistantAwareDrawer.Content>
    </AssistantAwareDrawer.Root>
  );
};
