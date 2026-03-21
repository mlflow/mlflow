import { first, isNil, isString } from 'lodash';
import { useEffect, useMemo, useState } from 'react';

import { Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import {
  KnownEvaluationResultAssessmentValueMapping,
  getEvaluationResultAssessmentValue,
  isDraftAssessment,
} from './GenAiEvaluationTracesReview.utils';
import type { RunEvaluationResultAssessment, RunEvaluationResultAssessmentDraft } from '../types';
import { timeSinceStr } from '../utils/DisplayUtils';
import { useMarkdownConverter } from '../utils/MarkdownUtils';

type AssessmentWithHistory = RunEvaluationResultAssessment | RunEvaluationResultAssessmentDraft;

export const EvaluationsReviewAssessmentDetailedHistory = ({
  history,
  alwaysExpanded = false,
}: {
  /**
   * List of assessments to display, ordered from the most recent to the oldest.
   */
  history: AssessmentWithHistory[];
  /**
   * Whether the detailed view is always expanded.
   */
  alwaysExpanded?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [referenceDate, setReferenceDate] = useState<Date>(new Date());

  useEffect(() => {
    const updateDateInterval = setInterval(() => {
      setReferenceDate(new Date());
    }, 5000);
    return () => {
      clearInterval(updateDateInterval);
    };
  }, []);

  const transitions = useMemo(() => {
    return history.reduce<[AssessmentWithHistory, AssessmentWithHistory][]>((result, next, index) => {
      const previous = history[index + 1];
      if (previous) {
        return [...result, [next, previous]];
      }

      return result;
    }, []);
  }, [history]);

  const { makeHTML } = useMarkdownConverter();

  const rationaleHtml = useMemo(() => {
    const rationale = first(history)?.rationale;
    return isString(rationale) ? makeHTML(rationale) : null;
  }, [history, makeHTML]);

  if (!transitions.length) {
    if (first(history)?.rationale) {
      return (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
          }}
        >
          {/* eslint-disable-next-line react/no-danger */}
          <span css={{ display: 'contents' }} dangerouslySetInnerHTML={{ __html: rationaleHtml ?? '' }} />
        </div>
      );
    } else if (first(history)?.errorMessage) {
      return (
        <>
          <Typography.Text color="error">{first(history)?.errorMessage}</Typography.Text>
        </>
      );
    } else {
      return (
        <>
          {!alwaysExpanded && (
            <div>
              <Spacer size="sm" />
              <Typography.Hint>
                <FormattedMessage
                  defaultMessage="No details for assessment"
                  description="Evaluation review > assessments > no history"
                />
              </Typography.Hint>
            </div>
          )}
        </>
      );
    }
  }

  const getMappedValue = (assessment: AssessmentWithHistory) => {
    const value = getEvaluationResultAssessmentValue(assessment);
    const knownMapping = KnownEvaluationResultAssessmentValueMapping[assessment.name];

    if (knownMapping && !isNil(value)) {
      const messageDescriptor = knownMapping[value.toString()] ?? knownMapping['default'];
      if (messageDescriptor) {
        return <FormattedMessage {...messageDescriptor} values={{ value }} />;
      }
    }

    return value;
  };

  return (
    <>
      <Spacer size="sm" />
      {transitions.map(([next, previous], index) => {
        const prevValue = getMappedValue(previous);
        const nextValue = getMappedValue(next);

        const isSameValue = prevValue === nextValue;

        const when = isDraftAssessment(next)
          ? intl.formatMessage({
              defaultMessage: 'just now',
              description: 'Evaluation review > assessments > tooltip > just now',
            })
          : timeSinceStr(next.timestamp, referenceDate);

        return (
          <div
            key={`${next.timestamp}-${index}`}
            css={{ marginBottom: !alwaysExpanded ? theme.spacing.md : undefined }}
          >
            <Typography.Hint css={{ marginBottom: theme.spacing.xs }}>
              {prevValue && nextValue && (
                <>
                  <code>{getMappedValue(previous)}</code> &#8594; <code>{getMappedValue(next)}</code>{' '}
                </>
              )}
              {isSameValue ? (
                <FormattedMessage
                  defaultMessage="added {when} by {user}"
                  description="Evaluation review > assessments > detailed history > added history entry"
                  values={{
                    when,
                    user: next.source?.sourceId,
                  }}
                />
              ) : (
                <FormattedMessage
                  defaultMessage="edited {when} by {user}"
                  description="Evaluation review > assessments > detailed history > edited history entry"
                  values={{
                    when,
                    user: next.source?.sourceId,
                  }}
                />
              )}
            </Typography.Hint>
            {next.rationale && <Typography.Text>{next.rationale}</Typography.Text>}
          </div>
        );
      })}
    </>
  );
};
