import { Button, FormUI, Input, useDesignSystemTheme } from '@databricks/design-system';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { v4 as uuidv4 } from 'uuid';

const AUTOSAVE_DELAY_MS = 500;

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
  const { theme } = useDesignSystemTheme();
  const errorId = useMemo(() => uuidv4(), []);
  const [draftName, setDraftName] = useState(initialName);
  const [saveError, setSaveError] = useState<'required' | 'request' | null>(null);
  const persistedNameRef = useRef(initialName);
  const latestDraftNameRef = useRef(initialName);
  const latestQueuedNameRef = useRef(initialName);
  const saveQueueRef = useRef(Promise.resolve());
  const autosaveTimerRef = useRef<ReturnType<typeof setTimeout>>();

  const queueSave = useCallback(
    (name: string) => {
      if (autosaveTimerRef.current) {
        clearTimeout(autosaveTimerRef.current);
      }
      if (!isEditable || name === persistedNameRef.current || name === latestQueuedNameRef.current) {
        return;
      }
      if (!name.trim()) {
        setSaveError('required');
        return;
      }

      latestQueuedNameRef.current = name;
      setSaveError(null);
      const pendingSave = saveQueueRef.current.then(async () => {
        if (name !== latestQueuedNameRef.current || name === persistedNameRef.current) {
          return;
        }
        try {
          await onSave(name);
          persistedNameRef.current = name;
          if (latestDraftNameRef.current === name) {
            setSaveError(null);
          }
        } catch {
          if (latestDraftNameRef.current === name) {
            setSaveError('request');
          }
        }
      });
      saveQueueRef.current = pendingSave.catch(() => undefined);
    },
    [isEditable, onSave],
  );

  useEffect(() => {
    latestDraftNameRef.current = draftName;
    if (!isEditable || draftName === persistedNameRef.current) {
      return;
    }
    autosaveTimerRef.current = setTimeout(() => queueSave(draftName), AUTOSAVE_DELAY_MS);
    return () => {
      if (autosaveTimerRef.current) {
        clearTimeout(autosaveTimerRef.current);
      }
    };
  }, [draftName, isEditable, queueSave]);

  useEffect(() => {
    if (latestDraftNameRef.current === persistedNameRef.current) {
      setDraftName(initialName);
      latestDraftNameRef.current = initialName;
    }
    persistedNameRef.current = initialName;
    latestQueuedNameRef.current = initialName;
  }, [initialName]);

  const retrySave = () => {
    latestQueuedNameRef.current = persistedNameRef.current;
    queueSave(latestDraftNameRef.current);
  };

  return (
    <div css={{ width: '40ch', maxWidth: '100%' }}>
      <Input
        componentId="mlflow.experiment_settings.name"
        aria-labelledby={labelId}
        aria-describedby={saveError ? errorId : undefined}
        aria-invalid={Boolean(saveError)}
        readOnly={!isEditable}
        validationState={saveError ? 'error' : undefined}
        value={draftName}
        onBlur={() => queueSave(latestDraftNameRef.current)}
        onChange={(event) => {
          setSaveError(null);
          setDraftName(event.target.value);
        }}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            event.currentTarget.blur();
          }
        }}
      />
      {saveError && (
        <div
          id={errorId}
          css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, marginBlockStart: theme.spacing.xs }}
        >
          <FormUI.Message
            type="error"
            message={
              saveError === 'required' ? (
                <FormattedMessage
                  defaultMessage="Experiment name is required."
                  description="Inline error shown when the experiment name is empty"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Unable to save the experiment name."
                  description="Inline error shown when an experiment rename fails"
                />
              )
            }
          />
          {saveError === 'request' && (
            <Button componentId="mlflow.experiment_settings.name_retry" type="link" onClick={retrySave}>
              <FormattedMessage defaultMessage="Retry" description="Link-styled button to retry an experiment rename" />
            </Button>
          )}
        </div>
      )}
    </div>
  );
};
