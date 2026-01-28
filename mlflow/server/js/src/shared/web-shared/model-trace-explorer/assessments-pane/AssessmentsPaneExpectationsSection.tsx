import { Button, PlusIcon, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { ExpectationAssessment } from '../ModelTrace.types';
import { ExpectationItem } from './ExpectationItem';
import { useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { isEmpty } from 'lodash';
import { AssessmentCreateForm } from './AssessmentCreateForm';

const AddExpectationButton = ({ onClick }: { onClick: () => void }) => (
  <Button componentId="shared.model-trace-explorer.add-expectation" size="small" icon={<PlusIcon />} onClick={onClick}>
    <FormattedMessage defaultMessage="Add expectation" description="Label for the button to add a new expectation" />
  </Button>
);

export const AssessmentsPaneExpectationsSection = ({
  expectations,
  activeSpanId,
  traceId,
}: {
  expectations: ExpectationAssessment[];
  activeSpanId?: string;
  traceId: string;
}) => {
  const sortedExpectations = useMemo(
    () => expectations.toSorted((left, right) => left.assessment_name.localeCompare(right.assessment_name)),
    [expectations],
  );

  const [createFormVisible, setCreateFormVisible] = useState(false);

  const { theme } = useDesignSystemTheme();
  return (
    <>
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          height: theme.spacing.lg,
          flexShrink: 0,
        }}
      >
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="Expectations"
            description="Label for the expectations section in the assessments pane"
          />{' '}
          {!isEmpty(sortedExpectations) && <>({sortedExpectations?.length})</>}
        </Typography.Text>
      </div>
      <div css={{ display: 'flex', justifyContent: 'flex-end', marginBottom: theme.spacing.sm }}>
        {!isEmpty(sortedExpectations) && <AddExpectationButton onClick={() => setCreateFormVisible(true)} />}
      </div>
      {sortedExpectations.length > 0 ? (
        <>
          <div
            css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, marginBottom: theme.spacing.sm }}
          >
            {sortedExpectations.map((expectation) => (
              <ExpectationItem expectation={expectation} key={expectation.assessment_id} />
            ))}
          </div>
        </>
      ) : (
        !createFormVisible && (
          <div
            css={{
              textAlign: 'center',
              borderRadius: theme.spacing.xs,
              border: `1px dashed ${theme.colors.border}`,
              padding: `${theme.spacing.md}px ${theme.spacing.md}px`,
            }}
          >
            <Typography.Hint>
              <FormattedMessage
                defaultMessage="Add a custom expectation to this trace."
                description="Hint message prompting user to add a new expectation"
              />{' '}
              <Typography.Link
                componentId="shared.model-trace-explorer.expectation-learn-more-link"
                openInNewTab
                href="https://www.mlflow.org/docs/latest/genai/assessments/expectations/"
              >
                <FormattedMessage
                  defaultMessage="Learn more."
                  description="Link text for learning more about expectations"
                />
              </Typography.Link>
            </Typography.Hint>
            <Spacer size="sm" />
            <AddExpectationButton onClick={() => setCreateFormVisible(true)} />
          </div>
        )
      )}
      {createFormVisible && (
        <AssessmentCreateForm
          spanId={activeSpanId}
          traceId={traceId}
          initialAssessmentType="expectation"
          setExpanded={() => setCreateFormVisible(false)}
        />
      )}
    </>
  );
};
