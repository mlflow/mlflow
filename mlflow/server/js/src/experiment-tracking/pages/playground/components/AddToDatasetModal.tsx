import { Alert, Checkbox, FormUI, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
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
  // Storing a reference answer is opt-in; unchecked, the record is created with inputs only.
  const [includeExpectedResponse, setIncludeExpectedResponse] = useState(false);
  const [submitError, setSubmitError] = useState<string | undefined>(undefined);

  const { selectedDatasets, hasMultiturnDataset, isCheckingMultiturn } = pickerState;

  const inputMessages = useMemo(() => getDatasetInputMessages(messages, variables), [messages, variables]);

  const {
    upsertDatasetRecordsMutationAsync,
    invalidateAfterUpsert,
    isLoading: isAdding,
  } = useUpsertDatasetRecordsMutation();

  // Bumped on cancel so a still-in-flight batch can tell it has been dismissed and skip
  // the success path (toast / closing) the user no longer asked for.
  const submissionGenerationRef = useRef(0);

  // Latest snapshot of the conversation for the open-time reset below, kept out of the
  // effect deps so editing messages while the modal is open doesn't overwrite the user's
  // edits to the expected response.
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  // Reset the form each time the modal is opened: clear the previous selection and search
  // (by remounting the picker), default the expected response to the latest assistant
  // reply, and drop any prior error or leftover in-flight batch. The opt-in starts checked
  // only when there is a reply to pre-fill, so the prompt-only case stays inputs-only.
  useEffect(() => {
    if (!visible) return;
    submissionGenerationRef.current += 1;
    setPickerResetKey((key) => key + 1);
    setPickerState(EMPTY_EVALUATION_DATASET_PICKER_STATE);
    const latestAssistantContent = getLatestAssistantContent(messagesRef.current);
    setExpectedResponse(latestAssistantContent);
    setIncludeExpectedResponse(latestAssistantContent.trim().length > 0);
    setSubmitError(undefined);
  }, [visible]);

  const hasInput = inputMessages.length > 0;
  const canAdd = selectedDatasets.length > 0 && hasInput && !hasMultiturnDataset && !isCheckingMultiturn && !isAdding;

  // One confirm fans out into one upsert per selected dataset. Record counts and the open
  // records table are invalidated for the datasets that succeeded, and a partial failure
  // surfaces as an error instead of a success toast.
  const handleAdd = async () => {
    if (!canAdd) return;
    setSubmitError(undefined);
    const generation = submissionGenerationRef.current;
    const datasets = selectedDatasets;
    const records = JSON.stringify([
      buildPlaygroundDatasetRecord({
        inputMessages,
        expectedResponse: includeExpectedResponse ? expectedResponse : '',
      }),
    ]);

    const results = await Promise.allSettled(
      datasets.map((dataset) => upsertDatasetRecordsMutationAsync({ datasetId: dataset.dataset_id, records })),
    );
    const succeeded = datasets.filter((_, index) => results[index].status === 'fulfilled');
    invalidateAfterUpsert(succeeded.map((dataset) => dataset.dataset_id));

    // The user dismissed the modal while the batch was in flight; the records still landed,
    // but the success toast / close would be for an interaction they already abandoned.
    if (generation !== submissionGenerationRef.current) return;

    const firstRejection = results.find((result) => result.status === 'rejected');
    if (firstRejection) {
      setSubmitError(
        getErrorMessage((firstRejection as PromiseRejectedResult).reason) ??
          intl.formatMessage({
            defaultMessage: 'Failed to add the record to the dataset',
            description: 'Fallback error shown when adding a playground prompt to an evaluation dataset fails',
          }),
      );
      return;
    }
    onAdded({ datasetNames: succeeded.map((dataset) => dataset.name) });
  };

  // Invalidate any in-flight batch on dismiss so a late upsert success can't fire the
  // onAdded success path after the user has cancelled.
  const handleCancel = () => {
    submissionGenerationRef.current += 1;
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
            defaultMessage="Save the current prompt as a record in an evaluation dataset. The messages become the record's inputs, and you can optionally store an expected response so you can score it with judges later."
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

        <div>
          <Checkbox
            componentId="mlflow.playground.add_to_dataset.include_expected_response"
            isChecked={includeExpectedResponse}
            onChange={setIncludeExpectedResponse}
          >
            <FormattedMessage
              defaultMessage="Add expected response"
              description="Label of the checkbox that opts into storing an expected response on the playground add-to-dataset modal"
            />
          </Checkbox>
          <FormUI.Hint>
            <FormattedMessage
              defaultMessage="Stored as expectations.expected_response — the reference answer that judges such as Correctness compare against."
              description="Hint under the expected-response editor on the playground add-to-dataset modal"
            />
          </FormUI.Hint>
          {includeExpectedResponse && (
            <TextArea
              componentId="mlflow.playground.add_to_dataset.expected_response"
              id="mlflow.playground.add_to_dataset.expected_response"
              value={expectedResponse}
              onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setExpectedResponse(event.target.value)}
              autoSize={{ minRows: 3, maxRows: 12 }}
              css={{ marginTop: theme.spacing.sm }}
              placeholder={intl.formatMessage({
                defaultMessage: 'The reference answer to score responses against',
                description: 'Placeholder for the expected-response editor on the playground add-to-dataset modal',
              })}
            />
          )}
        </div>
      </div>
    </Modal>
  );
};
