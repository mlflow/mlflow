import { Alert, FormUI, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  EMPTY_EVALUATION_DATASET_PICKER_STATE,
  EvaluationDatasetPicker,
} from '../../experiment-evaluation-datasets/components/EvaluationDatasetPicker';
import type { EvaluationDatasetPickerState } from '../../experiment-evaluation-datasets/components/EvaluationDatasetPicker';
import { useUpsertDatasetRecordsMutation } from '../../experiment-evaluation-datasets/hooks/useUpsertDatasetRecordsMutation';
import { buildPlaygroundDatasetRecord, getDatasetInputMessages, getLatestAssistantContent } from '../datasetRecord';
import type { ConversationMessage } from '../types';

const { TextArea } = Input;

const PREVIEW_CONTENT_CAP = 120;
const PICKER_TABLE_HEIGHT = 240;

const truncate = (s: string, cap: number) => (s.length > cap ? `${s.slice(0, cap)}…` : s);

// React Query types query/mutation errors as `unknown`; narrow to a displayable message.
const getErrorMessage = (error: unknown): string | undefined => (error instanceof Error ? error.message : undefined);

interface Props {
  visible: boolean;
  onCancel: () => void;
  experimentId: string;
  messages: ConversationMessage[];
  variables: Record<string, string>;
  onAdded: (result: { datasetNames: string[] }) => void;
}

export const AddToDatasetModal = ({ visible, onCancel, experimentId, messages, variables, onAdded }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [pickerState, setPickerState] = useState<EvaluationDatasetPickerState>(EMPTY_EVALUATION_DATASET_PICKER_STATE);
  // Remounting the picker resets its selection and search; bumped on every modal open.
  const [pickerResetKey, setPickerResetKey] = useState(0);
  const [expectedResponse, setExpectedResponse] = useState('');
  const [submitError, setSubmitError] = useState<string | undefined>(undefined);

  const { selectedDatasets, hasMultiturnDataset, isCheckingMultiturn } = pickerState;

  const inputMessages = useMemo(() => getDatasetInputMessages(messages, variables), [messages, variables]);

  // One confirm fans out into one upsert per selected dataset; notify the parent only after
  // the last one succeeds, and drop the batch on the first error so a partial failure
  // surfaces instead of a success toast.
  const pendingBatchRef = useRef<{ remaining: number; datasetNames: string[] } | null>(null);
  const { upsertDatasetRecordsMutation, isLoading: isAdding } = useUpsertDatasetRecordsMutation({
    onSuccess: () => {
      const batch = pendingBatchRef.current;
      if (!batch) return;
      batch.remaining -= 1;
      if (batch.remaining === 0) {
        pendingBatchRef.current = null;
        onAdded({ datasetNames: batch.datasetNames });
      }
    },
    onError: (error) => {
      pendingBatchRef.current = null;
      setSubmitError(
        getErrorMessage(error) ??
          intl.formatMessage({
            defaultMessage: 'Failed to add the record to the dataset',
            description: 'Fallback error shown when adding a playground prompt to an evaluation dataset fails',
          }),
      );
    },
  });

  // Latest snapshot of the conversation for the open-time reset below, kept out of the
  // effect deps so editing messages while the modal is open doesn't overwrite the user's
  // edits to the expected response.
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  // Reset the form each time the modal is opened: clear the previous selection and search
  // (by remounting the picker), default the expected response to the latest assistant
  // reply, and drop any prior error or leftover in-flight batch.
  useEffect(() => {
    if (!visible) return;
    pendingBatchRef.current = null;
    setPickerResetKey((key) => key + 1);
    setPickerState(EMPTY_EVALUATION_DATASET_PICKER_STATE);
    setExpectedResponse(getLatestAssistantContent(messagesRef.current));
    setSubmitError(undefined);
  }, [visible]);

  const hasInput = inputMessages.length > 0;
  const canAdd = selectedDatasets.length > 0 && hasInput && !hasMultiturnDataset && !isCheckingMultiturn && !isAdding;

  const handleAdd = () => {
    if (!canAdd) return;
    setSubmitError(undefined);
    const records = JSON.stringify([buildPlaygroundDatasetRecord({ inputMessages, expectedResponse })]);
    pendingBatchRef.current = {
      remaining: selectedDatasets.length,
      datasetNames: selectedDatasets.map((dataset) => dataset.name),
    };
    selectedDatasets.forEach((dataset) => upsertDatasetRecordsMutation({ datasetId: dataset.dataset_id, records }));
  };

  // Drop any in-flight batch on dismiss so a late upsert success can't fire the
  // onAdded success path after the user has cancelled.
  const handleCancel = () => {
    pendingBatchRef.current = null;
    onCancel();
  };

  return (
    <Modal
      componentId="mlflow.playground.add_to_dataset"
      visible={visible}
      onCancel={handleCancel}
      title={
        <FormattedMessage
          defaultMessage="Add to evaluation datasets"
          description="Title of the add-to-evaluation-datasets modal"
        />
      }
      okText={
        <FormattedMessage
          defaultMessage="{count, plural, =0 {Add to dataset} one {Add to dataset} other {Add to # datasets}}"
          description="Confirm-button label on the add-to-evaluation-datasets modal, reflecting how many datasets are selected"
          values={{ count: selectedDatasets.length }}
        />
      }
      okButtonProps={{ disabled: !canAdd, loading: isAdding }}
      onOk={handleAdd}
      cancelText={
        <FormattedMessage
          defaultMessage="Cancel"
          description="Cancel-button label on the playground add-to-evaluation-dataset modal"
        />
      }
      size="wide"
      zIndex={theme.options.zIndexBase + 10}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Paragraph withoutMargins>
          <FormattedMessage
            defaultMessage="Save the current prompt as a record in an evaluation dataset. The messages become the record's inputs, and the response is stored as the expected answer so you can score it with judges later."
            description="Intro paragraph at the top of the playground add-to-evaluation-dataset modal"
          />
        </Typography.Paragraph>

        {submitError && (
          <Alert
            componentId="mlflow.playground.add_to_dataset.error"
            type="error"
            closable={false}
            message={submitError}
          />
        )}

        {!hasInput && (
          <Alert
            componentId="mlflow.playground.add_to_dataset.no_input"
            type="warning"
            closable={false}
            message={
              <FormattedMessage
                defaultMessage="Add at least one non-empty message before adding this prompt to a dataset."
                description="Warning shown in the playground add-to-dataset modal when there is no input message to store"
              />
            }
          />
        )}

        <div>
          <EvaluationDatasetPicker
            key={pickerResetKey}
            experimentId={experimentId}
            onStateChange={setPickerState}
            tableHeight={PICKER_TABLE_HEIGHT}
            enabled={visible}
          />
        </div>

        <div>
          <FormUI.Label htmlFor="mlflow.playground.add_to_dataset.expected_response">
            <FormattedMessage
              defaultMessage="Expected response (optional)"
              description="Label for the expected-response editor on the playground add-to-dataset modal"
            />
          </FormUI.Label>
          <FormUI.Hint>
            <FormattedMessage
              defaultMessage="Stored as expectations.expected_response — the reference answer that judges such as Correctness compare against."
              description="Hint under the expected-response editor on the playground add-to-dataset modal"
            />
          </FormUI.Hint>
          <TextArea
            componentId="mlflow.playground.add_to_dataset.expected_response"
            id="mlflow.playground.add_to_dataset.expected_response"
            value={expectedResponse}
            onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setExpectedResponse(event.target.value)}
            autoSize={{ minRows: 3, maxRows: 12 }}
            placeholder={intl.formatMessage({
              defaultMessage: 'The reference answer to score responses against',
              description: 'Placeholder for the expected-response editor on the playground add-to-dataset modal',
            })}
          />
        </div>

        {hasInput && (
          <>
            <div css={{ borderTop: `1px solid ${theme.colors.border}` }} role="separator" aria-hidden="true" />
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.sm,
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.general.borderRadiusBase,
                padding: theme.spacing.md,
              }}
            >
              <Typography.Text
                size="sm"
                color="secondary"
                bold
                css={{ textTransform: 'uppercase', letterSpacing: 0.5 }}
              >
                <FormattedMessage
                  defaultMessage="Inputs"
                  description="Header of the input-messages preview in the playground add-to-dataset modal"
                />
              </Typography.Text>
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.xs,
                  maxHeight: 240,
                  overflowY: 'auto',
                  paddingRight: theme.spacing.xs,
                }}
              >
                {inputMessages.map((message, index) => (
                  <div
                    key={`${message.role}-${index}`}
                    css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'baseline' }}
                  >
                    <Typography.Text bold css={{ minWidth: 72 }}>
                      {message.role}
                    </Typography.Text>
                    <Typography.Text color="secondary">
                      {truncate(message.content ?? '', PREVIEW_CONTENT_CAP)}
                    </Typography.Text>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </Modal>
  );
};
