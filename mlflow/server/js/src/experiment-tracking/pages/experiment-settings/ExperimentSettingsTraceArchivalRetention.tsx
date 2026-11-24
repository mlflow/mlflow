import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { TraceArchivalRetentionInput } from '../../../common/components/TraceArchivalRetentionInput';
import {
  decodeTraceArchivalRetentionTag,
  encodeTraceArchivalRetentionTag,
  formatTraceArchivalRetention,
  getTraceArchivalRetentionValidationError,
  parseTraceArchivalRetention,
  TRACE_ARCHIVAL_RETENTION_TAG_KEY,
  type TraceArchivalRetentionUnit,
} from '../../../common/utils/traceArchival';
import { useTraceArchivalEnabled } from '../../hooks/useServerInfo';
import { MlflowService } from '../../sdk/MlflowService';
import type { ExperimentEntity } from '../../types';
import { ExperimentSettingsValue } from './ExperimentSettingsValue';
import { SettingCell, useElementId } from './SettingCell';
import { SettingSaveErrorMessage } from './SettingSaveErrorMessage';
import { useSettingAutosave } from './useSettingAutosave';

interface RetentionDraft {
  amount: string;
  unit: TraceArchivalRetentionUnit;
}

// A blank amount is the same value, no override, whatever unit is selected.
const retentionKey = ({ amount, unit }: RetentionDraft) => formatTraceArchivalRetention(amount, unit);

export interface ExperimentSettingsTraceArchivalRetentionProps {
  experiment: ExperimentEntity;
  refetchExperiment?: () => Promise<unknown>;
}

const TraceArchivalRetentionRow = ({
  experiment,
  refetchExperiment,
}: ExperimentSettingsTraceArchivalRetentionProps) => {
  const intl = useIntl();
  const amountInputId = useElementId('trace-archival-retention-amount');
  const unitSelectorId = useElementId('trace-archival-retention-unit');
  const errorId = useElementId('trace-archival-retention-error');
  const experimentId = experiment.experimentId;
  const savedRetention = decodeTraceArchivalRetentionTag(
    experiment.tags?.find((tag) => tag.key === TRACE_ARCHIVAL_RETENTION_TAG_KEY)?.value,
  );
  const savedValue = useMemo(() => parseTraceArchivalRetention(savedRetention), [savedRetention]);
  const { draft, updateDraft, saveNow, retry, error } = useSettingAutosave<RetentionDraft>({
    savedValue,
    getKey: retentionKey,
    validate: ({ amount, unit }) => getTraceArchivalRetentionValidationError(amount, unit, intl),
    save: async (value) => {
      const retention = retentionKey(value);
      if (retention) {
        await MlflowService.setExperimentTag({
          experiment_id: experimentId,
          key: TRACE_ARCHIVAL_RETENTION_TAG_KEY,
          value: encodeTraceArchivalRetentionTag(retention),
        });
      } else {
        await MlflowService.deleteExperimentTag({ experiment_id: experimentId, key: TRACE_ARCHIVAL_RETENTION_TAG_KEY });
      }
      // The retention has saved; a failed refresh afterwards isn't a failed save.
      await refetchExperiment?.()?.catch(() => undefined);
    },
  });

  return (
    <SettingCell.Control
      label={
        <FormattedMessage
          defaultMessage="Trace archival retention"
          description="Label for the experiment trace archival retention setting"
        />
      }
      description={
        <FormattedMessage
          defaultMessage="Optional. Set an experiment-specific archival retention override. Leave blank to inherit the workspace or server default."
          description="Description for the experiment trace archival retention setting"
        />
      }
    >
      {({ labelId }) => (
        <ExperimentSettingsValue>
          <div css={{ width: '40ch', maxWidth: '100%' }}>
            <TraceArchivalRetentionInput
              amount={draft.amount}
              amountInputId={amountInputId}
              amountAriaLabelledBy={labelId}
              amountAriaDescribedBy={error ? errorId : undefined}
              componentId="mlflow.experiment_settings.trace_archival_retention"
              error={Boolean(error)}
              onAmountBlur={() => saveNow()}
              onAmountChange={(amount) => updateDraft({ amount, unit: draft.unit })}
              onUnitChange={(unit) => {
                const next = { amount: draft.amount, unit };
                updateDraft(next);
                saveNow(next);
              }}
              unitSelectorId={unitSelectorId}
              unit={draft.unit}
            />
            {error && (
              <SettingSaveErrorMessage
                id={errorId}
                message={
                  // Like the experiment name: a refused change says why, and other failures get one message.
                  error.kind === 'invalid' ? (
                    error.message
                  ) : error.kind === 'permission' ? (
                    <FormattedMessage
                      defaultMessage="Unable to save the trace archival retention: {reason}"
                      description="Inline error shown when the server refuses a change to the experiment trace archival retention, with its reason, such as Permission denied"
                      values={{ reason: error.reason }}
                    />
                  ) : (
                    <FormattedMessage
                      defaultMessage="Unable to save the trace archival retention."
                      description="Inline error shown when saving the experiment trace archival retention fails"
                    />
                  )
                }
                onRetry={error.kind === 'invalid' ? undefined : retry}
                componentId="mlflow.experiment_settings.trace_archival_retention_retry"
              />
            )}
          </div>
        </ExperimentSettingsValue>
      )}
    </SettingCell.Control>
  );
};

/** The experiment's trace archival retention override; shown only when the server has trace archival on. */
export const ExperimentSettingsTraceArchivalRetention = (props: ExperimentSettingsTraceArchivalRetentionProps) => {
  const traceArchivalEnabled = useTraceArchivalEnabled();
  return traceArchivalEnabled ? <TraceArchivalRetentionRow {...props} /> : null;
};
