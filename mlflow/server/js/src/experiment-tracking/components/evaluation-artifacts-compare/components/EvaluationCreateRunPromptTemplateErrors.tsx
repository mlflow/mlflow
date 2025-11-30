import { Typography } from '@databricks/design-system';
import type { getPromptInputVariableNameViolations } from '../../prompt-engineering/PromptEngineering.utils';
import { FormattedMessage, defineMessage, useIntl } from 'react-intl';

const whitespaceViolationMessage = defineMessage({
  defaultMessage: 'The following variable names contain spaces which is disallowed: {invalidNames}',
  description: 'Experiment page > new run modal > variable name validation > including spaces error',
});

export const EvaluationCreateRunPromptTemplateErrors = ({
  violations,
}: {
  violations: ReturnType<typeof getPromptInputVariableNameViolations>;
}) => {
  const { namesWithSpaces } = violations;
  const { formatMessage } = useIntl();
  return (
    <>
      {namesWithSpaces.length > 0 && (
        <Typography.Text
          color="warning"
          size="sm"
          aria-label={formatMessage(whitespaceViolationMessage, {
            invalidNames: namesWithSpaces.join(', '),
          })}
        >
          <FormattedMessage
            {...whitespaceViolationMessage}
            values={{
              invalidNames: (
                <>
                  {namesWithSpaces.map((nameWithSpace) => (
                    <code key={nameWithSpace}>{nameWithSpace}</code>
                  ))}
                </>
              ),
            }}
          />
        </Typography.Text>
      )}
    </>
  );
};
