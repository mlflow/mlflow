// import {
//   Button,
//   ParagraphSkeleton,
//   PlusIcon,
//   Spacer,
//   TableSkeleton,
//   TableSkeletonRows,
//   Typography,
//   useDesignSystemTheme,
// } from '@databricks/design-system';
// import { FormattedMessage } from 'react-intl';
// import { useModelTraceExplorerUpdateTraceContext } from '../contexts/UpdateTraceContext';
// // import { useModelTraceExplorerUpdateTraceContext } from '../contexts/UpdateTraceContext';
// import { RunJudgeButtonForTrace } from '../../../../experiment-tracking/pages/experiment-scorers/RunJudgeButtonForTrace';
// import { FETCH_TRACE_INFO_QUERY_KEY } from '../ModelTraceExplorer.utils';
// import { useQueryClient } from '@tanstack/react-query';
// import { FeedbackAssessment } from '../ModelTrace.types';
// import { FeedbackGroup } from './FeedbackGroup';
// import { isEmpty } from 'lodash';
// import { Link } from '../RoutingUtils';
// import { invalidateMlflowSearchTracesCache } from '@databricks/web-shared/genai-traces-table';
// import { useState } from 'react';

// export const AssessmentJudgesSection = ({
//   traceId,
//   groupedFeedbacks,
// }: {
//   traceId: string;
//   groupedFeedbacks: [assessmentName: string, feedbacks: { [value: string]: FeedbackAssessment[] }][];
// }) => {
//   const { theme } = useDesignSystemTheme();
//   const queryClient = useQueryClient();
//   //   const [isLoading, setIsLoading] = useState(false);

//   const [scorerInProgressName, setScorerInProgressName] = useState<string | undefined>(undefined);

//   const handleScorerStarted = (scorerName: string) => {
//     setScorerInProgressName(scorerName);
//   };

//   const { useRenderRunJudgeButton, invalidateTraceQuery } = useModelTraceExplorerUpdateTraceContext();
//   if (!useRenderRunJudgeButton) {
//     return null;
//   }

//   return (
//     <div css={{ display: 'flex', flexDirection: 'column', marginBottom: theme.spacing.md }}>
//       <div
//         css={{
//           display: 'flex',
//           justifyContent: 'space-between',
//           alignItems: 'center',
//           marginBottom: theme.spacing.sm,
//           height: theme.spacing.lg,
//         }}
//       >
//         <Typography.Text bold>
//           <FormattedMessage
//             defaultMessage="Judges"
//             description="Label for the expectations section in the assessments pane"
//           />{' '}
//           {!isEmpty(groupedFeedbacks) && <>({groupedFeedbacks.length})</>}
//         </Typography.Text>
//         {!isEmpty(groupedFeedbacks) &&
//           null
//           // <RunJudgeButtonForTrace
//           //   traceId={traceId}
//           //   onScorerStarted={handleScorerStarted}
//           //   onScorerFinished={() => {
//           //     queryClient.invalidateQueries({ queryKey: [FETCH_TRACE_INFO_QUERY_KEY, traceId] });
//           //     invalidateMlflowSearchTracesCache({ queryClient });
//           //     invalidateTraceQuery?.(traceId);
//           //     setScorerInProgressName(undefined);
//           //   }}
//           // />
//         }
//       </div>
//       {groupedFeedbacks.map(([name, valuesMap]) => (
//         <FeedbackGroup key={name} name={name} valuesMap={valuesMap} traceId={traceId} showAddButton={false} />
//       ))}
//       {scorerInProgressName && (
//         <div
//           css={{
//             borderRadius: theme.spacing.sm,
//             border: `1px solid ${theme.colors.border}`,
//             padding: `${theme.spacing.md}px ${theme.spacing.md}px`,
//           }}
//         >
//           <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis', textWrap: 'nowrap' }}>
//             {scorerInProgressName}
//           </Typography.Text>
//           <Spacer size="sm" />
//           <TableSkeleton lines={3} />
//         </div>
//       )}
//       {isEmpty(groupedFeedbacks) && (
//         <div
//           css={{
//             textAlign: 'center',
//             borderRadius: theme.spacing.sm,
//             border: `1px solid ${theme.colors.border}`,
//             padding: `${theme.spacing.md}px ${theme.spacing.md}px`,
//             display: scorerInProgressName ? 'none' : 'block',
//           }}
//         >
//           <>
//             <Typography.Hint>
//               Add a LLM-as-a-judge or Custom code scorer to this trace.{' '}
//               <Typography.Link componentId="TODO">Learn more.</Typography.Link>
//             </Typography.Hint>
//             <Spacer size="sm" />
//             {/* {renderRunJudgeButton({ traceId })} */}

//             {/* <RunJudgeButtonForTrace
//               traceId={traceId}
//               onScorerStarted={handleScorerStarted}
//               onScorerFinished={() => {
//                 queryClient.invalidateQueries({ queryKey: [FETCH_TRACE_INFO_QUERY_KEY, traceId] });
//                 invalidateMlflowSearchTracesCache({ queryClient });
//                 invalidateTraceQuery?.(traceId);
//                 setScorerInProgressName(undefined);
//               }}
//             /> */}
//           </>
//         </div>
//       )}
//     </div>
//   );
// };
