import { first, groupBy, isEmpty, mapValues, orderBy } from 'lodash';
import { useMemo } from 'react';

import { Overflow, useDesignSystemTheme } from '@databricks/design-system';
import type { Assessment, ExpectationAssessment, ModelTrace } from '../ModelTrace.types';
import { isSessionLevelAssessment, isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';
import { AssessmentDisplayValue } from '../assessments-pane/AssessmentDisplayValue';
import { getAssessmentValue } from '../assessments-pane/utils';

const ASSESSMENT_DISPLAY_LIMIT = 5;

const isExpectationAssessment = (assessment: Assessment): assessment is ExpectationAssessment =>
  'expectation' in assessment;

export const SingleChatTurnAssessments = ({
  trace,
  getAssessmentTitle,
}: {
  trace: ModelTrace;
  getAssessmentTitle: (assessmentName: string) => string;
}) => {
  const { theme } = useDesignSystemTheme();
  const info = isV3ModelTraceInfo(trace.info) ? trace.info : null;

  const assessmentsToDisplay = useMemo(() => {
    // First, filter out session-level assessments
    const traceAssessments = (info?.assessments || []).filter((assessment) => !isSessionLevelAssessment(assessment));

    // Then, filter out invalid assessments
    const validAssessments = traceAssessments.filter((assessment) => assessment.valid !== false);

    // Group by assessment name
    const groups = groupBy(validAssessments, (assessment) => assessment.assessment_name);
    return mapValues(groups, (assessments) => orderBy(assessments, ['create_time'], ['desc']));
  }, [info?.assessments]);

  if (isEmpty(assessmentsToDisplay)) {
    return null;
  }

  const assessmentGroups = Object.values(assessmentsToDisplay);

  return (
    <div
      css={{
        display: 'flex',
        gap: theme.spacing.sm,
        flexWrap: 'wrap',
        width: '100%',
        minWidth: 0,
        overflow: 'hidden',
        marginTop: theme.spacing.sm,
      }}
    >
      <Overflow
        noMargin
        css={{ gap: theme.spacing.sm, width: '100%', minWidth: 0, maxWidth: '100%' }}
        visibleItemsCount={ASSESSMENT_DISPLAY_LIMIT}
      >
        {assessmentGroups.map((assessments) => {
          // In this preview, we only show the most recent assessment for each assessment name
          const assessment = first(assessments);
          if (!assessment) {
            return null;
          }
          const title = getAssessmentTitle(assessment.assessment_name);
          const value = getAssessmentValue(assessment)?.toString() ?? '';
          return (
            <div key={assessment.assessment_id}>
              {/* Both expectations and feedback assessments are formatted with title and value in the same box */}
              <AssessmentDisplayValue
                prefix={<>{title}: </>}
                jsonValue={value}
                css={{ maxWidth: 150 }}
                skipIcons={isExpectationAssessment(assessment)}
                overrideColor={isExpectationAssessment(assessment) ? 'default' : undefined}
              />
            </div>
          );
        })}
      </Overflow>
    </div>
  );
};
