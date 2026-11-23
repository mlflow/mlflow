import { Input } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useElementId } from './SettingCell';
import { SettingSaveErrorMessage } from './SettingSaveErrorMessage';
import { useSettingAutosave } from './useSettingAutosave';

// Names are saved trimmed, like the Edit experiment dialog does, so surrounding whitespace isn't a change.
const nameKey = (name: string) => name.trim();

export interface ExperimentSettingsNameInputProps {
  initialName: string;
  isEditable: boolean;
  labelId: string;
  onSave: (name: string) => Promise<void>;
}

export const ExperimentSettingsNameInput = ({
  initialName,
  isEditable,
  labelId,
  onSave,
}: ExperimentSettingsNameInputProps) => {
  const intl = useIntl();
  const errorId = useElementId('experiment-name-error');
  const { draft, updateDraft, saveNow, retry, error } = useSettingAutosave({
    savedValue: initialName,
    isEditable,
    getKey: nameKey,
    validate: (name) =>
      name.trim()
        ? undefined
        : intl.formatMessage({
            defaultMessage: 'Experiment name is required.',
            description: 'Inline error shown when the experiment name is empty',
          }),
    save: (name) => onSave(name.trim()),
  });

  return (
    <div css={{ width: '40ch', maxWidth: '100%' }}>
      <Input
        componentId="mlflow.experiment_settings.name"
        aria-labelledby={labelId}
        aria-describedby={error ? errorId : undefined}
        aria-invalid={Boolean(error)}
        disabled={!isEditable}
        validationState={error ? 'error' : undefined}
        value={draft}
        onBlur={() => {
          const trimmedName = draft.trim();
          if (trimmedName !== draft) {
            updateDraft(trimmedName);
          }
          saveNow(trimmedName);
        }}
        onChange={(event) => updateDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            event.currentTarget.blur();
          }
        }}
      />
      {error && (
        <SettingSaveErrorMessage
          id={errorId}
          message={
            error.kind === 'invalid' ? (
              error.message
            ) : error.reason ? (
              <FormattedMessage
                defaultMessage="Unable to save the experiment name: {reason}"
                description="Inline error shown when an experiment rename fails, with the server's reason"
                values={{ reason: error.reason }}
              />
            ) : (
              <FormattedMessage
                defaultMessage="Unable to save the experiment name."
                description="Inline error shown when an experiment rename fails"
              />
            )
          }
          onRetry={error.kind === 'request' ? retry : undefined}
          componentId="mlflow.experiment_settings.name_retry"
          retryDisabled={!isEditable}
        />
      )}
    </div>
  );
};
