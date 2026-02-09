import { Button } from '@databricks/design-system';

import { FormattedMessage } from 'react-intl';
import { useState } from 'react';
import { coerceToEnum } from '../../../shared/web-shared/utils';
import { SelectTracesModal } from '../../components/SelectTracesModal';
import { ScorerFormData } from './utils/scorerTransformUtils';
import { useFormContext } from 'react-hook-form';
import { MAX_SELECTED_ITEM_COUNT, ScorerEvaluationScope } from './constants';
import { SelectSessionsModal } from '../../components/SelectSessionsModal';

export const SampleScorerTracesToEvaluatePicker = ({
  selectedItemIds,
  onSelectedItemIdsChange,
}: {
  selectedItemIds: string[];
  onSelectedItemIdsChange: (selectedItemIds: string[]) => void;
}) => {
  const { watch } = useFormContext<ScorerFormData>();

  const evaluationScope = coerceToEnum(ScorerEvaluationScope, watch('evaluationScope'), ScorerEvaluationScope.TRACES);

  const [displayPickCustomTracesModal, setDisplayPickCustomTracesModal] = useState(false);
  const hasSelectedItems = selectedItemIds.length > 0;

  return (
    <>
      <Button
        componentId="mlflow.experiment-scorers.form.traces-picker.trigger"
        size="small"
        onClick={() => setDisplayPickCustomTracesModal(true)}
      >
        {hasSelectedItems ? (
          evaluationScope === ScorerEvaluationScope.TRACES ? (
            <FormattedMessage
              defaultMessage="{count, plural, one {1 trace selected} other {# traces selected}}"
              description="Label for the number of traces selected"
              values={{ count: selectedItemIds.length }}
            />
          ) : (
            <FormattedMessage
              defaultMessage="{count, plural, one {1 session selected} other {# sessions selected}}"
              description="Label for the number of sessions selected"
              values={{ count: selectedItemIds.length }}
            />
          )
        ) : evaluationScope === ScorerEvaluationScope.TRACES ? (
          <FormattedMessage defaultMessage="Select traces" description="Button to select traces" />
        ) : (
          <FormattedMessage defaultMessage="Select sessions" description="Button to select sessions" />
        )}
      </Button>
      {displayPickCustomTracesModal && evaluationScope === ScorerEvaluationScope.TRACES && (
        <SelectTracesModal
          onClose={() => {
            setDisplayPickCustomTracesModal(false);
          }}
          onSuccess={(traceIds) => {
            if (traceIds.length > 0) {
              onSelectedItemIdsChange(traceIds);
            }
            setDisplayPickCustomTracesModal(false);
          }}
          initialTraceIdsSelected={selectedItemIds}
          maxTraceCount={MAX_SELECTED_ITEM_COUNT}
        />
      )}
      {displayPickCustomTracesModal && evaluationScope === ScorerEvaluationScope.SESSIONS && (
        <SelectSessionsModal
          onClose={() => {
            setDisplayPickCustomTracesModal(false);
          }}
          onSuccess={(sessionIds) => {
            if (sessionIds.length > 0) {
              onSelectedItemIdsChange(sessionIds);
            }
            setDisplayPickCustomTracesModal(false);
          }}
          initialSessionIdsSelected={selectedItemIds}
          maxSessionCount={MAX_SELECTED_ITEM_COUNT}
        />
      )}
    </>
  );
};
