import { first, isNil } from 'lodash';
import { useMemo, useState } from 'react';

import { Spacer, Tag, TableSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl, FormattedMessage } from '@databricks/i18n';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import type { UseQueryResult } from '@databricks/web-shared/query-client';

import { EvaluationsReviewAssessments } from './EvaluationsReviewAssessments';
import { EvaluationsReviewListItemIndicator } from './EvaluationsReviewListItemIndicator';
import {
  isEvaluationResultReviewedAlready,
  KnownEvaluationResultAssessmentMetadataFields,
  getOrderedAssessments,
  KnownEvaluationRetrievalAssessmentNames,
  KnownEvaluationResultAssessmentName,
} from './GenAiEvaluationTracesReview.utils';
import { VerticalBar } from './VerticalBar';
import type {
  AssessmentInfo,
  RunEvaluationResultAssessmentDraft,
  RunEvaluationTracesDataEntry,
  RunEvaluationTracesRetrievalChunk,
} from '../types';
import { useMarkdownConverter } from '../utils/MarkdownUtils';
import { getRetrievedContextFromTrace } from '../utils/TraceUtils';

function isValidHttpUrl(str: any) {
  // The URL() constructor will throw on invalid URL
  try {
    const url = new URL(str);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch (err) {
    return false;
  }
}

const RetrievedChunkHeader = ({ chunk, index }: { chunk: RunEvaluationTracesRetrievalChunk; index: number }) => {
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
      }}
    >
      <Tag componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluations_components_evaluationsreviewretrievalsection.tsx_30">
        #{index + 1}
      </Tag>
      {isValidHttpUrl(chunk.docUrl) ? (
        <Typography.Link
          componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluations_components_evaluationsreviewretrievalsection.tsx_32"
          href={chunk.docUrl}
          ellipsis
          openInNewTab
          strong
        >
          {chunk.docUrl}
        </Typography.Link>
      ) : (
        <Typography.Title level={4} withoutMargins ellipsis>
          {chunk.docUrl}
        </Typography.Title>
      )}
    </div>
  );
};

const EvaluationsReviewSingleRunRetrievalSection = ({
  evaluationResult,
  onUpsertAssessment,
  overridingExistingReview = false,
  isReadOnly = false,
  assessmentInfos,
  traceQueryResult,
}: {
  evaluationResult: RunEvaluationTracesDataEntry;
  onUpsertAssessment: (assessment: RunEvaluationResultAssessmentDraft) => void;
  overridingExistingReview?: boolean;
  isReadOnly?: boolean;
  assessmentInfos: AssessmentInfo[];
  traceQueryResult: UseQueryResult<ModelTrace | undefined, unknown>;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [selectedIndex, setSelectedIndex] = useState<number>(0);

  const { makeHTML } = useMarkdownConverter();

  const evaluationRetrievalChunks = useMemo(() => {
    return !isNil(evaluationResult.retrievalChunks) && evaluationResult.retrievalChunks.length > 0
      ? evaluationResult.retrievalChunks
      : getRetrievedContextFromTrace(evaluationResult.responseAssessmentsByName, traceQueryResult.data);
  }, [evaluationResult.responseAssessmentsByName, evaluationResult.retrievalChunks, traceQueryResult.data]);

  const selectedEntryHtmlContent = useMemo(
    () => makeHTML(evaluationRetrievalChunks?.[selectedIndex]?.content),
    [evaluationRetrievalChunks, selectedIndex, makeHTML],
  );

  const noRetrievalFound = (evaluationRetrievalChunks || []).length === 0;

  const toBeReviewed =
    !isReadOnly && (!isEvaluationResultReviewedAlready(evaluationResult) || overridingExistingReview);

  const selectedChunk = evaluationRetrievalChunks?.[selectedIndex];

  const sectionTitle = intl.formatMessage({
    defaultMessage: 'Retrieval',
    description: 'Evaluation review > Retrieval section > title',
  });

  return (
    <div
      css={{
        paddingLeft: theme.spacing.md,
        paddingRight: theme.spacing.md,
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
      role="region"
      aria-label={sectionTitle}
    >
      <Typography.Text bold>{sectionTitle}</Typography.Text>

      {isNil(evaluationRetrievalChunks) && traceQueryResult.isFetching ? (
        <TableSkeleton lines={3} />
      ) : noRetrievalFound ? (
        <Typography.Text>
          <i>
            <FormattedMessage
              defaultMessage="No span with type RETRIEVER found in trace."
              description="GenAi Traces Table > Modal > Message displayed when no retrievals are found"
            />
          </i>
        </Typography.Text>
      ) : (
        <div
          css={{
            minHeight: 400,
            maxHeight: 600,
            overflow: 'hidden',
            display: 'flex',
            marginTop: theme.spacing.sm,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.general.borderRadiusBase,
          }}
        >
          <div
            css={{
              flex: 1,
              flexShrink: 1,
              maxWidth: 300,
              minWidth: 200,
              overflow: 'auto',
              padding: theme.spacing.sm,
              borderRight: `1px solid ${theme.colors.border}`,
            }}
            role="listbox"
          >
            {noRetrievalFound && (
              <Typography.Paragraph>
                <FormattedMessage
                  defaultMessage="No retrieval logged"
                  description="Evaluation review > retrieval section > no values"
                />
              </Typography.Paragraph>
            )}
            {(evaluationRetrievalChunks || []).map((chunk, index) => {
              const chunkRelevanceAssessmentInfo = assessmentInfos.find(
                (info) => info.name === KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE,
              );

              return (
                <div
                  role="option"
                  aria-label={chunk.content?.slice(0, 255)}
                  aria-selected={index === selectedIndex}
                  key={[chunk.docUrl, index].join('-')}
                  css={{
                    backgroundColor: index === selectedIndex ? theme.colors.actionIconBackgroundHover : 'transparent',
                    '&:hover': {
                      backgroundColor: theme.colors.actionIconBackgroundHover,
                    },
                    padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                    overflow: 'hidden',
                    display: 'flex',
                    gap: theme.spacing.sm,
                    alignItems: 'center',
                    cursor: 'pointer',
                    width: '100%',
                  }}
                  onClick={() => {
                    setSelectedIndex(index);
                  }}
                >
                  {/* TODO: Find a better way to determine which retrieval assessment to use for the indicator */}
                  <EvaluationsReviewListItemIndicator
                    chunkRelevanceAssessmentInfo={chunkRelevanceAssessmentInfo}
                    assessment={first(
                      chunk?.retrievalAssessmentsByName?.[KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE],
                    )}
                  />
                  <Typography.Text ellipsis css={{ flex: 1, lineHeight: theme.typography.lineHeightLg }}>
                    {chunk.content}
                  </Typography.Text>
                </div>
              );
            })}
          </div>

          {selectedChunk && (
            <div css={{ flex: 2, display: 'flex', flexDirection: 'column', width: 0 }}>
              <div
                css={{
                  padding: theme.spacing.md,
                  borderBottom: `1px solid ${theme.colors.border}`,
                }}
              >
                <RetrievedChunkHeader chunk={selectedChunk} index={selectedIndex} />
                <Spacer size="md" />
                <EvaluationsReviewAssessments
                  assessmentsType="retrieval"
                  assessmentsByName={getOrderedAssessments(selectedChunk.retrievalAssessmentsByName || {})}
                  onUpsertAssessment={(assessment: RunEvaluationResultAssessmentDraft) => {
                    if (!assessment.metadata) {
                      assessment.metadata = {};
                    }
                    // Set the chunk index to the selected index
                    assessment.metadata[KnownEvaluationResultAssessmentMetadataFields.CHUNK_INDEX] = selectedIndex;
                    onUpsertAssessment(assessment);
                  }}
                  allowEditing={toBeReviewed}
                  allowMoreThanOne
                  options={KnownEvaluationRetrievalAssessmentNames}
                  inputs={[selectedIndex]}
                  assessmentInfos={assessmentInfos}
                />
              </div>
              <div css={{ padding: theme.spacing.md, overflow: 'auto' }}>
                <Typography.Paragraph>
                  <span
                    css={{ display: 'contents' }}
                    // eslint-disable-next-line react/no-danger
                    dangerouslySetInnerHTML={{ __html: selectedEntryHtmlContent ?? '' }}
                  />
                </Typography.Paragraph>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

/**
 * Displays RAG retrieval results for a given evaluation result.
 */
export const EvaluationsReviewRetrievalSection = ({
  evaluationResult,
  otherEvaluationResult,
  onUpsertAssessment,
  overridingExistingReview = false,
  isReadOnly = false,
  assessmentInfos,
  traceQueryResult,
  compareToTraceQueryResult,
}: {
  evaluationResult?: RunEvaluationTracesDataEntry;
  otherEvaluationResult?: RunEvaluationTracesDataEntry;
  onUpsertAssessment: (assessment: RunEvaluationResultAssessmentDraft) => void;
  overridingExistingReview?: boolean;
  isReadOnly?: boolean;
  assessmentInfos: AssessmentInfo[];
  traceQueryResult: UseQueryResult<ModelTrace | undefined, unknown>;
  compareToTraceQueryResult: UseQueryResult<ModelTrace | undefined, unknown>;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        width: '100%',
        gap: theme.spacing.sm,
      }}
    >
      {evaluationResult && (
        <EvaluationsReviewSingleRunRetrievalSection
          evaluationResult={evaluationResult}
          onUpsertAssessment={onUpsertAssessment}
          overridingExistingReview={overridingExistingReview}
          isReadOnly={isReadOnly}
          assessmentInfos={assessmentInfos}
          traceQueryResult={traceQueryResult}
        />
      )}
      {otherEvaluationResult && (
        <>
          <VerticalBar />
          <EvaluationsReviewSingleRunRetrievalSection
            evaluationResult={otherEvaluationResult}
            onUpsertAssessment={onUpsertAssessment}
            overridingExistingReview={overridingExistingReview}
            isReadOnly={isReadOnly}
            assessmentInfos={assessmentInfos}
            traceQueryResult={compareToTraceQueryResult}
          />
        </>
      )}
    </div>
  );
};
