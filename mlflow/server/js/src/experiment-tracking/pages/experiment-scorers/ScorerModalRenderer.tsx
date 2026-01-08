import React from 'react';
import { Modal } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { isRunningScorersEnabled } from '../../../common/utils/FeatureUtils';
import ScorerFormCreateContainer from './ScorerFormCreateContainer';
import ScorerFormEditContainer from './ScorerFormEditContainer';
import { COMPONENT_ID_PREFIX, SCORER_FORM_MODE, type ScorerFormMode } from './constants';
import type { ScheduledScorer } from './types';
import type { ScorerFormData } from './utils/scorerTransformUtils';

interface ScorerModalRendererProps {
  experimentId: string;
  visible: boolean;
  onClose: () => void;
  mode: ScorerFormMode;
  existingScorer?: ScheduledScorer;
  initialScorerType?: ScorerFormData['scorerType'];
}

const ScorerModalRenderer: React.FC<ScorerModalRendererProps> = ({
  experimentId,
  visible,
  onClose,
  mode,
  existingScorer,
  initialScorerType,
}) => {
  const isRunningScorersFeatureEnabled = isRunningScorersEnabled();

  return (
    <Modal
      componentId={`${COMPONENT_ID_PREFIX}.scorer-modal`}
      title={
        mode === SCORER_FORM_MODE.EDIT ? (
          <FormattedMessage defaultMessage="Edit judge" description="Title for edit judge modal" />
        ) : initialScorerType === 'custom-code' ? (
          <FormattedMessage
            defaultMessage="Create custom code judge"
            description="Title for new custom code judge modal"
          />
        ) : (
          <FormattedMessage defaultMessage="Create LLM judge" description="Title for new LLM judge modal" />
        )
      }
      visible={visible}
      onCancel={onClose}
      footer={null}
      destroyOnClose
      {...(initialScorerType !== 'custom-code' && {
        size: 'wide' as const,
        css: {
          width: '100% !important',
        },
      })}
      {...(isRunningScorersFeatureEnabled &&
        initialScorerType !== 'custom-code' && {
          verticalSizing: 'maxed_out' as const,
          dangerouslySetAntdProps: {
            bodyStyle: {
              display: 'flex',
              flexDirection: 'column',
              flex: 1,
              minHeight: 0,
              overflow: 'hidden',
            },
          },
        })}
    >
      {mode === SCORER_FORM_MODE.EDIT && existingScorer ? (
        <ScorerFormEditContainer experimentId={experimentId} onClose={onClose} existingScorer={existingScorer} />
      ) : (
        <ScorerFormCreateContainer
          experimentId={experimentId}
          onClose={onClose}
          initialScorerType={initialScorerType}
        />
      )}
    </Modal>
  );
};

export default ScorerModalRenderer;
