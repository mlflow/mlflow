import { isEmpty } from 'lodash';

import { Button, importantify, PlusIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { Assessment, ExpectationAssessment, ModelTrace } from '../ModelTrace.types';
import { isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';
import { AssessmentDisplayValue } from '../assessments-pane/AssessmentDisplayValue';
import { getAssessmentValue } from '../assessments-pane/utils';

const isExpectationAssessment = (assessment: Assessment): assessment is ExpectationAssessment =>
  'expectation' in assessment;

export const SingleChatTurnAssessments = ({
  trace,
  getAssessmentTitle,
  onAddAssessmentsClick,
}: {
  trace: ModelTrace;
  getAssessmentTitle: (assessmentName: string) => string;
  onAddAssessmentsClick?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const info = isV3ModelTraceInfo(trace.info) ? trace.info : null;

  if (!info?.assessments || isEmpty(info?.assessments)) {
    return (
      <div css={{ display: 'flex', justifyContent: 'flex-start' }}>
        <Button
          componentId="shared.model-trace-explorer.session-view.single-chart-turn-assessments.add-assessment"
          icon={<PlusIcon />}
          size="small"
          onClick={onAddAssessmentsClick}
          css={[{ marginTop: theme.spacing.sm }, importantify({ backgroundColor: theme.colors.backgroundPrimary })]}
        >
          <FormattedMessage
            defaultMessage="Evaluate trace"
            description="A call to action button to add assessments to a model trace in the single chat turn view"
          />
        </Button>
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', gap: theme.spacing.sm, flexWrap: 'wrap' }}>
      {info?.assessments.map((assessment, index) => {
        const title = getAssessmentTitle(assessment.assessment_name);
        const value = getAssessmentValue(assessment)?.toString() ?? '';
        return (
          <div key={assessment.assessment_id}>
            {/* Expectations assessments are formatted differently than feedback assessments */}
            {isExpectationAssessment(assessment) ? (
              <AssessmentDisplayValue
                prefix={<>{title}: </>}
                jsonValue={value}
                css={{ maxWidth: 150 }}
                skipIcons
                overrideColor="default"
                assessmentName={assessment.assessment_name}
              />
            ) : (
              <>
                {title}:{' '}
                <AssessmentDisplayValue
                  jsonValue={value}
                  css={{ maxWidth: 150 }}
                  assessmentName={assessment.assessment_name}
                />
              </>
            )}
          </div>
        );
      })}
    </div>
  );
};
