import { FormUI, SimpleSelect, SimpleSelectOption, Spacer } from '@databricks/design-system';
import { useFormContext } from 'react-hook-form';
import { ScorerFormData } from './utils/scorerTransformUtils';
import { FormattedMessage } from 'react-intl';

export const ScorerFormEvaluationScopeSelect = () => {
  const { register } = useFormContext<ScorerFormData>();

  return (
    <>
      <FormUI.Label>
        <FormattedMessage defaultMessage="Select scope" description="Label for the scorer evaluation scope selection" />
      </FormUI.Label>
      <FormUI.Hint>
        <FormattedMessage
          defaultMessage="What do you want the scorer to evaluate?"
          description="Hint for the scorer evaluation scope selection"
        />
      </FormUI.Hint>
      <SimpleSelect
        componentId="mlflow.experiment-scorers.form.scope-select"
        id="mlflow.experiment-scorers.form.scope-select"
        css={{ width: '100%' }}
        {...register('evaluationScope')}
      >
        <SimpleSelectOption value="traces">Traces</SimpleSelectOption>
        <SimpleSelectOption value="sessions">Sessions</SimpleSelectOption>
      </SimpleSelect>
      <Spacer />
    </>
  );
};
