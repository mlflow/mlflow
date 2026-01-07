import React from 'react';
import { Modal } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { isRunningScorersEnabled } from '../../../common/utils/FeatureUtils';
import ScorerFormCreateContainer from './ScorerFormCreateContainer';
import ScorerFormEditContainer from './ScorerFormEditContainer';
import { COMPONENT_ID_PREFIX, SCORER_FORM_MODE, type ScorerFormMode } from './constants';
import type { ScheduledScorer } from './types';

interface ScorerModalRendererProps {
  experimentId: string;
  visible: boolean;
  onClose: () => void;
  mode: ScorerFormMode;
  existingScorer?: ScheduledScorer;
}

const ScorerModalRenderer: React.FC<ScorerModalRendererProps> = ({
  experimentId,
  visible,
  onClose,
  mode,
  existingScorer,
}) => {
  const isRunningScorersFeatureEnabled = isRunningScorersEnabled();

  return (
    <Modal
      componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorermodalrenderer_29"
      title={
        mode === SCORER_FORM_MODE.EDIT ? (
          <FormattedMessage defaultMessage="Edit judge" description="Title for edit judge modal" />
        ) : (
          <FormattedMessage defaultMessage="Create judge" description="Title for new judge modal" />
        )
      }
      visible={visible}
      onCancel={onClose}
      footer={null}
      destroyOnClose
      size="wide"
      css={{
        width: '100% !important',
      }}
      {...(isRunningScorersFeatureEnabled && {
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
        <ScorerFormCreateContainer experimentId={experimentId} onClose={onClose} />
      )}
    </Modal>
  );
};

export default ScorerModalRenderer;
