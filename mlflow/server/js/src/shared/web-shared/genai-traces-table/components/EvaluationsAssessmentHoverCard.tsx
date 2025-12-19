import {
  BinaryIcon,
  BracketsXIcon,
  NumbersIcon,
  SparkleDoubleIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
  UserIcon,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { EvaluationsRcaStats } from './EvaluationsRcaStats';
import { KnownEvaluationResultAssessmentName } from '../enum';
import type { AssessmentAggregates, AssessmentFilter, AssessmentInfo, AssessmentValueType } from '../types';

export const EvaluationsAssessmentHoverCard = ({
  assessmentInfo,
  assessmentNameToAggregates,
  allAssessmentFilters,
  toggleAssessmentFilter,
  runUuid,
  compareToRunUuid,
}: {
  assessmentInfo: AssessmentInfo;
  assessmentNameToAggregates: Record<string, AssessmentAggregates>;
  allAssessmentFilters: AssessmentFilter[];
  toggleAssessmentFilter: (
    assessmentName: string,
    filterValue: AssessmentValueType,
    run: string,
    filterType?: AssessmentFilter['filterType'],
  ) => void;
  runUuid?: string;
  compareToRunUuid?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  return (
    <>
      <div
        css={{
          maxWidth: '25rem',
          display: 'flex',
          flexDirection: 'column',
          overflowWrap: 'break-word',
          wordBreak: 'break-word',
          gap: theme.spacing.sm,
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
          }}
        >
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.xs,
              paddingBottom: theme.spacing.sm,
              borderBottom: `1px solid ${theme.colors.border}`,
            }}
          >
            <div
              css={{
                display: 'flex',
                gap: theme.spacing.xs,
                alignItems: 'center',
              }}
            >
              {/* Dtype icon  */}
              {assessmentInfo.dtype === 'numeric' ? (
                <NumbersIcon />
              ) : assessmentInfo.dtype === 'boolean' ? (
                <BinaryIcon />
              ) : (
                <></>
              )}

              <Typography.Title
                level={4}
                css={{
                  marginBottom: 0,
                  marginRight: theme.spacing.xs,
                }}
              >
                {assessmentInfo.displayName}
              </Typography.Title>
              {assessmentInfo.source?.sourceType === 'AI_JUDGE' && (
                <Tag color="turquoise" componentId="mlflow.experiment.evaluations.ai-judge-tag">
                  <div
                    css={{
                      display: 'flex',
                      gap: theme.spacing.xs,
                    }}
                  >
                    <SparkleDoubleIcon
                      css={{
                        color: theme.colors.textSecondary,
                      }}
                    />
                    <Typography.Hint>
                      {intl.formatMessage({
                        defaultMessage: 'AI Judge',
                        description: 'Label for AI judge in the tooltip for the assessment in the evaluation metrics.',
                      })}
                    </Typography.Hint>
                  </div>
                </Tag>
              )}
              {assessmentInfo.source?.sourceType === 'HUMAN' && (
                <Tag color="coral" componentId="mlflow.experiment.evaluations.human-judge-tag">
                  <div
                    css={{
                      display: 'flex',
                      gap: theme.spacing.xs,
                    }}
                  >
                    <UserIcon />
                    <Typography.Hint>
                      {intl.formatMessage({
                        defaultMessage: 'Human judge',
                        description:
                          'Label for human judge in the tooltip for the assessment in the evaluation metrics.',
                      })}
                    </Typography.Hint>
                  </div>
                </Tag>
              )}
              {assessmentInfo.isCustomMetric && (
                <Tag color="indigo" componentId="mlflow.experiment.evaluations.ai-judge-tag">
                  <div
                    css={{
                      display: 'flex',
                      gap: theme.spacing.xs,
                    }}
                  >
                    <Typography.Hint>
                      <BracketsXIcon />
                    </Typography.Hint>

                    <Typography.Hint>{assessmentInfo.metricName}</Typography.Hint>
                  </div>
                </Tag>
              )}
            </div>
            <div>
              <Typography.Hint>{assessmentInfo.name}</Typography.Hint>
            </div>
          </div>
        </div>
        <div>
          <span
            css={{
              display: 'contents',
              '& p': {
                marginBottom: 0,
              },
            }}
          >
            {assessmentInfo.description}
          </span>
        </div>
        {assessmentInfo.name === KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT && runUuid ? (
          <div>
            <EvaluationsRcaStats
              overallAssessmentInfo={assessmentInfo}
              assessmentNameToAggregates={assessmentNameToAggregates}
              allAssessmentFilters={allAssessmentFilters}
              toggleAssessmentFilter={toggleAssessmentFilter}
              runUuid={runUuid}
              compareToRunUuid={compareToRunUuid}
            />
          </div>
        ) : (
          <></>
        )}
      </div>
    </>
  );
};
