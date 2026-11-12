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
import { buildPlaygroundDatasetRecord, getDatasetInputMessages, getInputTurns, getLatestReply } from '../datasetRecord';
import type { ChatMessage, ConversationMessage } from '../types';
import { getEmptyVariables } from '../utils';

const { TextArea } = Input;

const PREVIEW_CONTENT_CAP = 120;
const PICKER_TABLE_HEIGHT = 240;

const truncate = (s: string, cap: number) => (s.length > cap ? `${s.slice(0, cap)}…` : s);

const getToolCallNames = (message: ChatMessage) =>
  (message.tool_calls ?? []).flatMap((toolCall) => (toolCall.function?.name ? [toolCall.function.name] : []));

// React Query types query/mutation errors as `unknown`; narrow to a displayable message.
const getErrorMessage = (error: unknown): string | undefined => (error instanceof Error ? error.message : undefined);

export interface AddedToDatasetResult {
  datasetId: string;
  name: string;
  /** True when the dataset already had a record with these inputs, so that record was updated. */
  updated: boolean;
}

interface Props {
  visible: boolean;
  onCancel: () => void;
  experimentId: string;
  messages: ConversationMessage[];
  variables: Record<string, string>;
  /** The registry prompt version loaded in the playground, linked from the record when set. */
  linkedPrompt?: { name: string; version: string } | null;
  onAdded: (result: { datasets: AddedToDatasetResult[] }) => void;
}

export const AddToDatasetModal = ({
  visible,
  onCancel,
  experimentId,
  messages,
  variables,
  linkedPrompt,
  onAdded,
}: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [pickerState, setPickerState] = useState<EvaluationDatasetPickerState>(EMPTY_EVALUATION_DATASET_PICKER_STATE);
  // Remounting the picker resets its selection and search; bumped on every modal open.
  const [pickerResetKey, setPickerResetKey] = useState(0);
  const [expectedResponse, setExpectedResponse] = useState('');
  // Storing a reference answer is opt-in; unchecked, the record is created with inputs only.
  const [includeExpectedResponse, setIncludeExpectedResponse] = useState(false);
  const [submitError, setSubmitError] = useState<string | undefined>(undefined);
  // The upsert cannot be cancelled once sent, so the modal stays open until it settles.
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { selectedDatasets, hasMultiturnDataset, isCheckingMultiturn } = pickerState;

  const inputMessages = useMemo(() => getDatasetInputMessages(messages, variables), [messages, variables]);
  const emptyVariables = useMemo(() => getEmptyVariables(getInputTurns(messages), variables), [messages, variables]);
  const latestReply = useMemo(() => getLatestReply(messages, variables), [messages, variables]);

  const { upsertDatasetRecordsMutationAsync, invalidateAfterUpsert } = useUpsertDatasetRecordsMutation();

  // Read by the open-time reset below, kept out of its deps so a change while the modal is
  // open doesn't overwrite the user's edits to the expected response.
  const latestReplyRef = useRef(latestReply);
  latestReplyRef.current = latestReply;

  // Reset the form each time the modal is opened: clear the previous selection and search
  // (by remounting the picker), default the expected response to the latest reply, and drop
  // any prior error. The opt-in starts checked only when there is a reply to pre-fill, so
  // the prompt-only case stays inputs-only.
  useEffect(() => {
    if (!visible) return;
    setPickerResetKey((key) => key + 1);
    setPickerState(EMPTY_EVALUATION_DATASET_PICKER_STATE);
    const { content } = latestReplyRef.current;
    setExpectedResponse(content);
    setIncludeExpectedResponse(content.trim().length > 0);
    setSubmitError(undefined);
  }, [visible]);

  const hasInput = inputMessages.length > 0;
  const canAdd =
    selectedDatasets.length > 0 &&
    hasInput &&
    emptyVariables.length === 0 &&
    !hasMultiturnDataset &&
    !isCheckingMultiturn &&
    !isSubmitting;

  // One confirm fans out into one upsert per selected dataset. Record counts and the open
  // records table are invalidated for the datasets that succeeded, and a partial failure
  // surfaces as an error instead of a success toast.
  const handleAdd = async () => {
    if (!canAdd) return;
    setSubmitError(undefined);
    setIsSubmitting(true);
    const datasets = selectedDatasets;
    const records = JSON.stringify([
      buildPlaygroundDatasetRecord({
        inputMessages,
        expectedResponse: includeExpectedResponse ? expectedResponse : '',
        linkedPrompt,
      }),
    ]);

    try {
      const results = await Promise.allSettled(
        datasets.map((dataset) => upsertDatasetRecordsMutationAsync({ datasetId: dataset.dataset_id, records })),
      );
      invalidateAfterUpsert(
        datasets.filter((_, index) => results[index].status === 'fulfilled').map((dataset) => dataset.dataset_id),
      );

      const firstRejection = results.find((result): result is PromiseRejectedResult => result.status === 'rejected');
      if (firstRejection) {
        setSubmitError(
          getErrorMessage(firstRejection.reason) ??
            intl.formatMessage({
              defaultMessage: 'Failed to add the record to the dataset',
              description: 'Fallback error shown when adding a playground prompt to an evaluation dataset fails',
            }),
        );
        return;
      }
      onAdded({
        datasets: datasets.map((dataset, index) => {
          const result = results[index];
          return {
            datasetId: dataset.dataset_id,
            name: dataset.name,
            updated: result.status === 'fulfilled' && (result.value.updated_count ?? 0) > 0,
          };
        }),
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal
      componentId="mlflow.playground.add_to_dataset"
      visible={visible}
      onCancel={isSubmitting ? undefined : onCancel}
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
      okButtonProps={{ disabled: !canAdd, loading: isSubmitting }}
      onOk={handleAdd}
      cancelText={
        <FormattedMessage
          defaultMessage="Cancel"
          description="Cancel-button label on the playground add-to-evaluation-dataset modal"
        />
      }
      cancelButtonProps={{ disabled: isSubmitting }}
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

        {emptyVariables.length > 0 && (
          <Alert
            componentId="mlflow.playground.add_to_dataset.empty_variables"
            type="warning"
            closable={false}
            message={
              <FormattedMessage
                defaultMessage="Provide a value for {count, plural, one {the variable} other {the variables}} {names} before adding this prompt to a dataset."
                description="Warning shown in the playground add-to-dataset modal when a prompt variable has no value, which blocks adding the prompt"
                values={{ count: emptyVariables.length, names: emptyVariables.join(', ') }}
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
                      {message.content?.trim() ? (
                        truncate(message.content, PREVIEW_CONTENT_CAP)
                      ) : (
                        <FormattedMessage
                          defaultMessage="Tool call: {names}"
                          description="Preview of an assistant turn in the playground add-to-dataset modal that only calls tools; lists the called tool names"
                          values={{ names: getToolCallNames(message).join(', ') }}
                        />
                      )}
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
          {/* One hint: the checkbox only indents the hint that directly follows it. */}
          <FormUI.Hint>
            <FormattedMessage
              defaultMessage="Stored as expectations.expected_response — the reference answer that judges such as Correctness compare against."
              description="Hint under the expected-response editor on the playground add-to-dataset modal"
            />
            {latestReply.isOutdated && (
              <>
                <br />
                <FormattedMessage
                  defaultMessage="The prompt changed after the last run, so its reply is not pre-filled. Run the prompt again to pre-fill it."
                  description="Hint on the playground add-to-dataset modal when the latest reply was generated from different inputs than the current prompt"
                />
              </>
            )}
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
