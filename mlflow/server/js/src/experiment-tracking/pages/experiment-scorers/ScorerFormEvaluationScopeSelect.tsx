import { FormUI, Radio, RHFControlledComponents, Spacer } from '@databricks/design-system';
import { useFormContext } from 'react-hook-form';
import { ScorerFormData } from './utils/scorerTransformUtils';
import { FormattedMessage } from 'react-intl';
import { SCORER_FORM_MODE, ScorerEvaluationScope, ScorerFormMode } from './constants';

export const ScorerFormEvaluationScopeSelect = ({ mode }: { mode: ScorerFormMode }) => {
  const { control } = useFormContext<ScorerFormData>();

  const isEditMode = mode === SCORER_FORM_MODE.EDIT;

  return (
    <>
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
      <RHFControlledComponents.RadioGroup
        componentId="mlflow.experiment-scorers.form.scope-select"
        id="mlflow.experiment-scorers.form.scope-select"
        control={control}
        name="evaluationScope"
        disabled={isEditMode}
      >
        <Radio value={ScorerEvaluationScope.TRACES}>
          <FormattedMessage defaultMessage="Traces" description="Label for the scorer evaluation scope selection" />
          <FormUI.Hint>
            <FormattedMessage
              defaultMessage="Use a large language model to automatically evaluate traces."
              description="Hint for the scorer evaluation scope selection for traces"
            />
          </FormUI.Hint>
        </Radio>
        <Radio value={ScorerEvaluationScope.SESSIONS}>
          <FormattedMessage defaultMessage="Sessions" description="Label for the scorer evaluation scope selection" />
          <FormUI.Hint>
            <FormattedMessage
              defaultMessage="Evaluate entire sessions for conversation quality and outcomes."
              description="Hint for the scorer evaluation scope selection for sessions"
            />
          </FormUI.Hint>
        </Radio>
      </RHFControlledComponents.RadioGroup>
      <Spacer />
    </>
  );
};
