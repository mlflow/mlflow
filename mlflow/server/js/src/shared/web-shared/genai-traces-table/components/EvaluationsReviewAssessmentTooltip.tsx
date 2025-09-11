import { first } from 'lodash';
import { useEffect, useMemo, useState } from 'react';

import { Tooltip } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import {
  KnownEvaluationResultAssessmentValueMapping,
  getEvaluationResultAssessmentValue,
  isDraftAssessment,
  isEvaluationResultOverallAssessment,
  hasBeenEditedByHuman,
} from './GenAiEvaluationTracesReview.utils';
import type { RunEvaluationResultAssessment } from '../types';
import { timeSinceStr } from '../utils/DisplayUtils';

export const EvaluationsReviewAssessmentTooltip = ({
  assessmentHistory,
  children,
  disable = false,
}: React.PropsWithChildren<{
  assessmentHistory: RunEvaluationResultAssessment[];
  disable?: boolean;
}>) => {
  const intl = useIntl();

  const isOverallAssessment = useMemo(
    () => assessmentHistory.some(isEvaluationResultOverallAssessment),
    [assessmentHistory],
  );

  const [referenceDate, setReferenceDate] = useState<Date>(new Date());

  useEffect(() => {
    const updateDateInterval = setInterval(() => {
      setReferenceDate(new Date());
    }, 5000);
    return () => {
      clearInterval(updateDateInterval);
    };
  }, []);

  const getTitle = () => {
    const mostRecentEntry = first(assessmentHistory);
    if (!mostRecentEntry) {
      return undefined;
    }
    const isEditedByHuman = hasBeenEditedByHuman(mostRecentEntry);

    if (isEditedByHuman) {
      const previousRecentEntry = assessmentHistory[1];
      const previousRecentValue = previousRecentEntry
        ? getEvaluationResultAssessmentValue(previousRecentEntry)?.toString()
        : undefined;

      const timeSince = isDraftAssessment(mostRecentEntry)
        ? intl.formatMessage({
            defaultMessage: 'just now',
            description: 'Evaluation review > assessments > tooltip > just now',
          })
        : timeSinceStr(mostRecentEntry.timestamp, referenceDate);

      if (previousRecentValue) {
        const mappedValue = KnownEvaluationResultAssessmentValueMapping[mostRecentEntry.name]?.[previousRecentValue];
        const displayedPreviousValue = mappedValue ? intl.formatMessage(mappedValue) : previousRecentValue;

        return (
          <FormattedMessage
            defaultMessage="Edited {timeSince} by {source}. Original value: {value}"
            values={{
              timeSince,
              source: mostRecentEntry?.source?.sourceId,
              value: displayedPreviousValue,
            }}
            description="Evaluation review > assessments > tooltip > edited by human"
          />
        );
      }

      return (
        <FormattedMessage
          defaultMessage="Edited {timeSince} by {source}."
          values={{
            timeSince,
            source: mostRecentEntry?.source?.sourceId,
          }}
          description="Evaluation review > assessments > tooltip > edited by human"
        />
      );
    }

    if (isOverallAssessment) {
      return (
        <FormattedMessage
          defaultMessage="Overall assessment added by LLM-as-a-judge"
          description="Evaluation review > assessments > tooltip > overall assessment added by LLM-as-a-judge"
        />
      );
    }

    if (mostRecentEntry?.errorMessage) {
      return mostRecentEntry?.errorMessage;
    }

    return (
      <FormattedMessage
        defaultMessage="Assessment added by LLM-as-a-judge"
        description="Evaluation review > assessments > tooltip > assessment added by LLM-as-a-judge"
      />
    );
  };
  return (
    <Tooltip
      componentId="web-shared.genai-traces-table.evaluations-review-assessment.tooltip"
      content={disable ? undefined : <span>{getTitle()}</span>}
      side="top"
    >
      {children}
    </Tooltip>
  );
};
