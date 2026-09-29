import React from 'react';
import { Controller, useWatch, type Control, type UseFormSetValue } from 'react-hook-form';
import {
  FormUI,
  Input,
  Typography,
  useDesignSystemTheme,
  DialogCombobox,
  DialogComboboxTrigger,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
} from '@databricks/design-system';
import { FormattedMessage, useIntl, type IntlShape } from '@databricks/i18n';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { SCORER_FORM_MODE, type ScorerFormMode, type ScorerEvaluationScope } from './constants';
import type { JevAnswerType } from './types';
import type { ScorerFormData } from './utils/scorerTransformUtils';
import { validateJevCriteria, type JevCriteriaError } from './utils/jevScorerUtils';
import { ModelSectionRenderer } from './ModelSectionRenderer';
import EvaluateTracesSection from './EvaluateTracesSection';

export interface JevScorerFormData {
  scorerType: 'jev';
  name: string;
  model: string;
  question: string;
  answerType: JevAnswerType;
  criteria: string;
  threshold: string;
  sampleRate: number;
  filterString?: string;
  evaluationScope?: ScorerEvaluationScope;
}

interface JevScorerFormRendererProps {
  mode: ScorerFormMode;
  control: Control<JevScorerFormData>;
  setValue: UseFormSetValue<JevScorerFormData>;
}

const criteriaErrorMessage = (error: JevCriteriaError, intl: IntlShape): string => {
  switch (error) {
    case 'json':
      return intl.formatMessage({
        defaultMessage: 'Enter valid JSON for the selected answer type.',
        description: 'Invalid Jev criteria JSON',
      });
    case 'score':
      return intl.formatMessage({
        defaultMessage: 'Enter a JSON array of 2 to 10 text descriptions.',
        description: 'Invalid Jev score criteria',
      });
    case 'object':
      return intl.formatMessage({
        defaultMessage: 'Enter a JSON object mapping answers to descriptions.',
        description: 'Invalid Jev object criteria',
      });
    case 'descriptions':
      return intl.formatMessage({
        defaultMessage: 'Each answer must have a non-empty label and a text description.',
        description: 'Invalid Jev criteria description',
      });
    case 'noul':
      return intl.formatMessage({
        defaultMessage: 'Noul criteria can only describe true and false.',
        description: 'Invalid Jev noul criteria',
      });
    case 'choice':
      return intl.formatMessage({
        defaultMessage: 'Enter between 1 and 255 choices.',
        description: 'Invalid Jev choice criteria',
      });
  }
};

const DIRECT_SDK_EXAMPLE = `import mlflow
from mlflow.genai.scorers import make_jev_scorer

# Set TYPESAFE_API_KEY in your environment.
judge = make_jev_scorer(
    name="answer_relevance",
    model="typesafe:/jev-latest",
    question="Does the answer address the user's question?",
    answer_type="noul",
    threshold=0.7,
)
mlflow.genai.evaluate(
    data=[{"inputs": {"question": "What is 2 + 2?"}, "outputs": "4"}],
    scorers=[judge],
)`;

const JevScorerFormRenderer: React.FC<JevScorerFormRendererProps> = ({ mode, control, setValue }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const answerType = useWatch({ control, name: 'answerType' });
  const isReadOnly = mode === SCORER_FORM_MODE.DISPLAY;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {!isReadOnly && (
        <div>
          <FormUI.Label htmlFor="jev-scorer-name" required>
            <FormattedMessage defaultMessage="Name" description="Name of a Jev scorer" />
          </FormUI.Label>
          <Controller
            name="name"
            control={control}
            rules={{
              validate: (value) =>
                Boolean(value.trim()) ||
                intl.formatMessage({
                  defaultMessage: 'This field is required.',
                  description: 'Missing required Jev scorer field',
                }),
            }}
            render={({ field, fieldState }) => (
              <>
                <Input
                  {...field}
                  id="jev-scorer-name"
                  componentId="mlflow.experiment-scorers.jev-name"
                  disabled={mode !== SCORER_FORM_MODE.CREATE}
                />
                {fieldState.error && <FormUI.Message type="error" message={fieldState.error.message} />}
              </>
            )}
          />
        </div>
      )}
      <ModelSectionRenderer
        mode={mode}
        control={control as Control<ScorerFormData>}
        setValue={setValue as UseFormSetValue<ScorerFormData>}
        provider="typesafe"
        allowDirectModel={false}
      />
      <FormUI.Hint>
        <FormattedMessage
          defaultMessage="Choose a TypeSafe endpoint. Its API key is managed in AI Gateway."
          description="Gateway credentials guidance for Jev scorers"
        />
      </FormUI.Hint>
      <div>
        <FormUI.Label htmlFor="jev-scorer-question" required>
          <FormattedMessage defaultMessage="Question" description="Evaluation question for a Jev scorer" />
        </FormUI.Label>
        <FormUI.Hint>
          <FormattedMessage
            defaultMessage="Ask one question about the response. Inputs, outputs, and any expectations are supplied as context automatically."
            description="Context sent to a Jev scorer"
          />
        </FormUI.Hint>
        <Controller
          name="question"
          control={control}
          rules={{
            validate: (value) =>
              Boolean(value.trim()) ||
              intl.formatMessage({
                defaultMessage: 'This field is required.',
                description: 'Missing required Jev scorer field',
              }),
          }}
          render={({ field, fieldState }) => (
            <>
              <Input.TextArea
                {...field}
                id="jev-scorer-question"
                componentId="mlflow.experiment-scorers.jev-question"
                readOnly={isReadOnly}
                rows={3}
              />
              {fieldState.error && <FormUI.Message type="error" message={fieldState.error.message} />}
            </>
          )}
        />
      </div>
      <div>
        <FormUI.Label htmlFor="jev-scorer-answer-type" required>
          <FormattedMessage defaultMessage="Answer type" description="Primitive answer type for a Jev scorer" />
        </FormUI.Label>
        <Controller
          name="answerType"
          control={control}
          rules={{ required: true }}
          render={({ field }) => (
            <DialogCombobox
              id="jev-scorer-answer-type"
              componentId="mlflow.experiment-scorers.jev-answer-type"
              value={[field.value]}
            >
              <DialogComboboxTrigger
                withInlineLabel={false}
                allowClear={false}
                disabled={isReadOnly}
                renderDisplayedValue={(value) =>
                  value === 'noul'
                    ? intl.formatMessage({ defaultMessage: 'Noul (probability)', description: 'Noul answer type' })
                    : value === 'choice'
                      ? intl.formatMessage({ defaultMessage: 'Choice', description: 'Choice answer type' })
                      : intl.formatMessage({ defaultMessage: 'Score', description: 'Score answer type' })
                }
              />
              {!isReadOnly && (
                <DialogComboboxContent>
                  <DialogComboboxOptionList>
                    {(['noul', 'choice', 'score'] as const).map((value) => (
                      <DialogComboboxOptionListSelectItem
                        key={value}
                        value={value}
                        checked={field.value === value}
                        onChange={() => {
                          field.onChange(value);
                          setValue('criteria', '', { shouldValidate: true, shouldDirty: true });
                          setValue('threshold', '', { shouldValidate: true, shouldDirty: true });
                        }}
                      >
                        {value === 'noul' ? (
                          <FormattedMessage defaultMessage="Noul (probability)" description="Noul answer type" />
                        ) : value === 'choice' ? (
                          <FormattedMessage defaultMessage="Choice" description="Choice answer type" />
                        ) : (
                          <FormattedMessage defaultMessage="Score" description="Score answer type" />
                        )}
                      </DialogComboboxOptionListSelectItem>
                    ))}
                  </DialogComboboxOptionList>
                </DialogComboboxContent>
              )}
            </DialogCombobox>
          )}
        />
      </div>
      <div>
        <FormUI.Label htmlFor="jev-scorer-criteria" required={answerType !== 'noul'}>
          <FormattedMessage defaultMessage="Criteria" description="Answer criteria for a Jev scorer" />
        </FormUI.Label>
        <FormUI.Hint>
          {answerType === 'score' ? (
            <FormattedMessage
              defaultMessage="Enter a JSON array of 2 to 10 ordered level descriptions. Scores range from 0 to the last level's index and may be fractional."
              description="Score criteria entry guidance"
            />
          ) : answerType === 'choice' ? (
            <FormattedMessage
              defaultMessage="Enter a JSON object mapping each answer label to its description (up to 255 choices)."
              description="Choice criteria entry guidance"
            />
          ) : (
            <FormattedMessage
              defaultMessage="Optionally enter a JSON object describing true and false. Noul returns the probability that the answer is true."
              description="Noul criteria entry guidance"
            />
          )}
        </FormUI.Hint>
        <Controller
          name="criteria"
          control={control}
          rules={{
            validate: (value) => {
              const result = validateJevCriteria(answerType, value);
              return result === true || criteriaErrorMessage(result, intl);
            },
          }}
          render={({ field, fieldState }) => (
            <>
              <Input.TextArea
                {...field}
                id="jev-scorer-criteria"
                componentId="mlflow.experiment-scorers.jev-criteria"
                readOnly={isReadOnly}
                rows={4}
                placeholder={
                  answerType === 'score'
                    ? '["Incorrect", "Partially correct", "Correct"]'
                    : answerType === 'choice'
                      ? '{"billing": "Payment questions", "technical": "Product questions"}'
                      : '{"true": "Addresses the question", "false": "Misses the question"}'
                }
              />
              {fieldState.error && <FormUI.Message type="error" message={fieldState.error.message} />}
            </>
          )}
        />
      </div>
      {answerType === 'noul' && (
        <div>
          <FormUI.Label htmlFor="jev-scorer-threshold">
            <FormattedMessage
              defaultMessage="Threshold"
              description="Optional probability threshold for a Jev scorer"
            />
          </FormUI.Label>
          <FormUI.Hint>
            <FormattedMessage
              defaultMessage="Leave blank to return a probability. Set a threshold from 0 to 1 to return true when the probability meets or exceeds it. The probability is retained with the result."
              description="Noul threshold behavior"
            />
          </FormUI.Hint>
          <Controller
            name="threshold"
            control={control}
            rules={{
              validate: (value) =>
                value === '' ||
                (Number.isFinite(Number(value)) && Number(value) >= 0 && Number(value) <= 1) ||
                intl.formatMessage({
                  defaultMessage: 'Enter a number between 0 and 1.',
                  description: 'Invalid Jev threshold',
                }),
            }}
            render={({ field, fieldState }) => (
              <>
                <Input
                  {...field}
                  id="jev-scorer-threshold"
                  componentId="mlflow.experiment-scorers.jev-threshold"
                  type="number"
                  min={0}
                  max={1}
                  step="any"
                  readOnly={isReadOnly}
                />
                {fieldState.error && <FormUI.Message type="error" message={fieldState.error.message} />}
              </>
            )}
          />
        </div>
      )}
      <EvaluateTracesSection control={control} mode={mode} />
      {!isReadOnly && (
        <div>
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Use Jev from the SDK" description="Jev direct SDK example heading" />
          </Typography.Title>
          <Typography.Paragraph>
            <FormattedMessage
              defaultMessage="For local evaluation without a gateway endpoint, configure TYPESAFE_API_KEY and use a typesafe:/ model. Saved judges and automatic evaluation use gateway endpoints."
              description="Direct Jev SDK usage guidance"
            />
          </Typography.Paragraph>
          <CodeSnippet language="python" theme={theme.isDarkMode ? 'duotoneDark' : 'light'}>
            {DIRECT_SDK_EXAMPLE}
          </CodeSnippet>
        </div>
      )}
    </div>
  );
};

export default JevScorerFormRenderer;
