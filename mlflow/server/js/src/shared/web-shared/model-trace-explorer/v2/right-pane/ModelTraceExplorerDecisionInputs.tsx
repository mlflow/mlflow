import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { DecisionInputField, DecisionQuestionViewModel } from '../../decision/Decision.types';
import { DecisionQuestionsRenderer } from '../../decision/DecisionQuestionsRenderer';
import { resolveDecisionInputFields } from '../../decision/resolveDecisionInputFields';
import type { Assessment } from '../ModelTrace.types';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';

interface InputField {
  key: string;
  value: string;
}

const assertNever = (_value: never): never => {
  throw new Error('Unhandled decision input field');
};

const DecisionQuestionsField = ({
  title,
  questions,
}: {
  title: string;
  questions: readonly DecisionQuestionViewModel[];
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ position: 'relative' }}>
      {title && (
        <div
          css={{
            alignItems: 'center',
            display: 'flex',
            justifyContent: 'space-between',
            marginBottom: theme.spacing.xs,
            paddingInline: 0,
          }}
        >
          <Typography.Text bold color="secondary" size="sm">
            {title}
          </Typography.Text>
        </div>
      )}
      <DecisionQuestionsRenderer questions={questions} />
    </div>
  );
};

export const ModelTraceExplorerDecisionInputs = ({
  fields,
  viewModels,
  assessments,
}: {
  fields: readonly InputField[];
  viewModels: readonly DecisionInputField[];
  assessments?: Assessment[];
}) => {
  const { theme } = useDesignSystemTheme();
  const resolvedFields = resolveDecisionInputFields(fields, viewModels);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {resolvedFields.map(({ field, viewModel }, index) => {
        switch (viewModel.kind) {
          case 'field':
            return (
              <ModelTraceExplorerFieldRenderer
                key={field.key || index}
                title={field.key}
                data={field.value}
                renderMode="default"
                assessments={assessments}
              />
            );
          case 'questions':
            return <DecisionQuestionsField key={field.key || index} title={field.key} questions={viewModel.items} />;
          default:
            return assertNever(viewModel);
        }
      })}
    </div>
  );
};
