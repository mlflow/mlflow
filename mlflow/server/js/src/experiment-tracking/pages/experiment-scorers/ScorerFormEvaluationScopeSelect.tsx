import { FormUI, Radio } from '@databricks/design-system';
import type { RadioChangeEvent } from 'antd';
import { Controller, useFormContext } from 'react-hook-form';
import { ScorerFormData } from './utils/scorerTransformUtils';
import { FormattedMessage } from 'react-intl';
import { SCORER_FORM_MODE, ScorerEvaluationScope, ScorerFormMode } from './constants';
import type { LLMScorerFormData } from './LLMScorerFormRenderer';

interface ScorerFormEvaluationScopeSelectProps {
  mode: ScorerFormMode;
  onUserSelect?: (fieldName: keyof LLMScorerFormData, value: string) => void;
}

export const ScorerFormEvaluationScopeSelect = ({ mode, onUserSelect }: ScorerFormEvaluationScopeSelectProps) => {
  const { control } = useFormContext<ScorerFormData>();

  const isEditMode = mode === SCORER_FORM_MODE.EDIT;

  return (
    <div>
      <FormUI.Label>
        <FormattedMessage
          defaultMessage="Select scope"
          description="Label for the scorer evaluation scope/level selection (either traces or sessions)"
        />
      </FormUI.Label>
      <FormUI.Hint>
        <FormattedMessage
          defaultMessage="What do you want the scorer to evaluate?"
          description="Hint for the scorer evaluation scope selection"
        />
      </FormUI.Hint>
      <Controller
        name="evaluationScope"
        control={control}
        render={({ field }) => (
          <Radio.Group
            name="evaluationScope"
            componentId="mlflow.experiment-scorers.form.scope-select"
            value={field.value}
            onChange={(e: RadioChangeEvent) => {
              const newValue = e.target.value;
              field.onChange(newValue);
              onUserSelect?.('evaluationScope', newValue);
            }}
            disabled={isEditMode}
          >
            <Radio value={ScorerEvaluationScope.TRACES}>
              <FormattedMessage defaultMessage="Traces" description="Label for the scorer evaluation scope selection" />
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage="Evaluate individual traces for quality and correctness."
                  description="Hint for the scorer evaluation scope selection for traces"
                />
              </FormUI.Hint>
            </Radio>
            <Radio value={ScorerEvaluationScope.SESSIONS}>
              <FormattedMessage
                defaultMessage="Sessions"
                description="Label for the scorer evaluation scope selection"
              />
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage="Evaluate entire sessions for conversation quality and outcomes."
                  description="Hint for the scorer evaluation scope selection for sessions"
                />
              </FormUI.Hint>
            </Radio>
          </Radio.Group>
        )}
      />
    </div>
  );
};
