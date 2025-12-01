import { useMemo, useState } from 'react';
import {
  useDesignSystemTheme,
  LegacySkeleton,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { Data as PlotlyData, Layout } from 'plotly.js';
import { LazyPlot } from '../../../LazyPlot';

type TraceGroupBy = 'trace_name' | 'trace_status';
type TokenGroupBy = 'trace_name' | 'token_type';
type AssessmentGroupBy = 'assessment_name' | 'assessment_type';

interface TracesStatisticsProps {
  experimentIds: string[];
  timeRange?: { startTime: string | undefined; endTime: string | undefined };
}

interface TraceGroup {
  name: string;
  count: number;
}

interface TokenGroup {
  name: string;
  count: number;
}

interface AssessmentGroup {
  name: string;
  count: number;
  avg?: number;
  count0?: number;
  count1?: number;
  type: 'boolean' | 'numeric' | 'string';
}

interface StatisticCardProps {
  title: string;
  value: number | string;
  isLoading?: boolean;
}

const StatisticCard = ({ title, value, isLoading }: StatisticCardProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        flex: 1,
        padding: theme.spacing.md,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        minWidth: 0,
      }}
    >
      <div
        css={{
          fontSize: theme.typography.fontSizeSm,
          color: theme.colors.textSecondary,
          fontWeight: theme.typography.typographyRegularFontWeight,
        }}
      >
        {title}
      </div>
      {isLoading ? (
        <LegacySkeleton active paragraph={false} />
      ) : (
        <div
          css={{
            fontSize: theme.typography.fontSizeXxl,
            fontWeight: theme.typography.typographyBoldFontWeight,
            color: theme.colors.textPrimary,
          }}
        >
          {typeof value === 'number' ? value.toLocaleString() : value}
        </div>
      )}
    </div>
  );
};

export const TracesStatistics = ({ experimentIds, timeRange }: TracesStatisticsProps) => {
  const { theme } = useDesignSystemTheme();
  const [traceGroupBy, setTraceGroupBy] = useState<TraceGroupBy>('trace_name');
  const [tokenGroupBy, setTokenGroupBy] = useState<TokenGroupBy>('trace_name');
  const [assessmentGroupBy, setAssessmentGroupBy] = useState<AssessmentGroupBy>('assessment_name');

  // TODO: Replace with actual API calls when backend is ready
  // The real implementation would use something like:
  // const { data, isLoading } = useQuery({
  //   queryKey: ['traceStatistics', experimentIds, timeRange?.startTime, timeRange?.endTime, traceGroupBy],
  //   queryFn: async () => {
  //     const response = await MlflowService.queryTraceMetrics({
  //       experiment_ids: experimentIds,
  //       view_type: 'TRACES',
  //       measure: 'trace',
  //       aggregation_type: ['COUNT'],
  //       dimensions: [traceGroupBy], // Group by trace_name or trace_status
  //     });
  //     return response;
  //   },
  // });

  // For now, using mock data
  const isLoading = false;

  const statistics = useMemo(() => {
    // Mock data - generate realistic numbers based on time range
    const startTimeMs = timeRange?.startTime ? parseInt(timeRange.startTime, 10) : Date.now() - 7 * 24 * 60 * 60 * 1000;
    const endTimeMs = timeRange?.endTime ? parseInt(timeRange.endTime, 10) : Date.now();
    const timeRangeDays = (endTimeMs - startTimeMs) / (24 * 60 * 60 * 1000);
    
    // Scale mock data based on time range
    const dailyTraces = 150;
    const tokensPerTrace = 2500;
    const assessmentRate = 0.7; // 70% of traces have assessments
    
    const totalTraces = Math.floor(dailyTraces * timeRangeDays);
    const totalTokens = Math.floor(totalTraces * tokensPerTrace);
    const totalAssessments = Math.floor(totalTraces * assessmentRate);
    
    return {
      totalTraces,
      totalTokens,
      totalAssessments,
    };
  }, [experimentIds, timeRange]);

  // Generate mock grouped trace data
  const groupedTraces = useMemo((): TraceGroup[] => {
    const totalTraces = statistics.totalTraces;
    
    if (traceGroupBy === 'trace_name') {
      // Mock trace names with realistic distribution
      const traceNames = [
        'chat_completion',
        'text_generation', 
        'embedding_generation',
        'question_answering',
        'summarization',
        'code_generation',
      ];
      
      // Generate counts that sum to totalTraces
      const counts = traceNames.map((_, i) => {
        // Create a realistic distribution (some traces are more common)
        const weight = Math.pow(0.7, i); // Exponential decay
        return weight;
      });
      
      const totalWeight = counts.reduce((sum, w) => sum + w, 0);
      
      return traceNames.map((name, i) => ({
        name,
        count: Math.floor((counts[i] / totalWeight) * totalTraces),
      }));
    } else {
      // Mock trace statuses
      return [
        { name: 'OK', count: Math.floor(totalTraces * 0.85) },
        { name: 'ERROR', count: Math.floor(totalTraces * 0.10) },
        { name: 'IN_PROGRESS', count: Math.floor(totalTraces * 0.05) },
      ];
    }
  }, [traceGroupBy, statistics.totalTraces]);

  // Create horizontal bar chart data for grouped traces
  const groupedTracesPlotData: PlotlyData[] = useMemo(() => {
    if (groupedTraces.length === 0) return [];

    return [
      {
        y: groupedTraces.map((_, i) => i),
        x: groupedTraces.map(g => g.count),
        type: 'bar',
        orientation: 'h',
        marker: {
          color: 'rgba(1, 148, 226, 0.7)',
        },
        text: groupedTraces.map(g => g.name),
        textposition: 'inside',
        textfont: {
          color: 'white',
          size: 12,
        },
        insidetextanchor: 'start',
        hovertemplate: '<b>%{text}</b>: %{x} traces<extra></extra>',
      },
    ];
  }, [groupedTraces]);

  const groupedTracesLayout: Partial<Layout> = useMemo(() => ({
    height: Math.max(200, groupedTraces.length * 40 + 60),
    margin: { l: 20, r: 20, t: 10, b: 40 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    autosize: true,
    xaxis: {
      title: 'Count',
      showgrid: true,
      gridcolor: theme.colors.grey200,
      color: theme.colors.textPrimary,
    },
    yaxis: {
      showgrid: false,
      color: theme.colors.textPrimary,
      showticklabels: false,
      automargin: true,
    },
    font: {
      family: theme.typography.fontFamily,
      color: theme.colors.textPrimary,
    },
  }), [groupedTraces.length, theme]);

  // Generate mock grouped token data
  const groupedTokens = useMemo((): TokenGroup[] => {
    const totalTokens = statistics.totalTokens;
    
    if (tokenGroupBy === 'trace_name') {
      // Use the same trace names as in the traces chart
      const traceNames = [
        'chat_completion',
        'text_generation', 
        'embedding_generation',
        'question_answering',
        'summarization',
        'code_generation',
      ];
      
      // Generate counts that sum to totalTokens
      const counts = traceNames.map((_, i) => {
        const weight = Math.pow(0.7, i); // Exponential decay
        return weight;
      });
      
      const totalWeight = counts.reduce((sum, w) => sum + w, 0);
      
      return traceNames.map((name, i) => ({
        name,
        count: Math.floor((counts[i] / totalWeight) * totalTokens),
      }));
    } else {
      // Mock token types (input vs output)
      return [
        { name: 'Input Tokens', count: Math.floor(totalTokens * 0.40) },
        { name: 'Output Tokens', count: Math.floor(totalTokens * 0.60) },
      ];
    }
  }, [tokenGroupBy, statistics.totalTokens]);

  // Create horizontal bar chart data for grouped tokens
  const groupedTokensPlotData: PlotlyData[] = useMemo(() => {
    if (groupedTokens.length === 0) return [];

    return [
      {
        y: groupedTokens.map((_, i) => i),
        x: groupedTokens.map(g => g.count),
        type: 'bar',
        orientation: 'h',
        marker: {
          color: 'rgba(1, 148, 226, 0.7)',
        },
        text: groupedTokens.map(g => g.name),
        textposition: 'inside',
        textfont: {
          color: 'white',
          size: 12,
        },
        insidetextanchor: 'start',
        hovertemplate: '<b>%{text}</b>: %{x} tokens<extra></extra>',
      },
    ];
  }, [groupedTokens]);

  const groupedTokensLayout: Partial<Layout> = useMemo(() => ({
    height: Math.max(200, groupedTokens.length * 40 + 60),
    margin: { l: 20, r: 20, t: 10, b: 40 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    autosize: true,
    xaxis: {
      title: 'Count',
      showgrid: true,
      gridcolor: theme.colors.grey200,
      color: theme.colors.textPrimary,
    },
    yaxis: {
      showgrid: false,
      color: theme.colors.textPrimary,
      showticklabels: false,
      automargin: true,
    },
    font: {
      family: theme.typography.fontFamily,
      color: theme.colors.textPrimary,
    },
  }), [groupedTokens.length, theme]);

  // Generate mock grouped assessment data
  const groupedAssessments = useMemo((): AssessmentGroup[] => {
    // Mock assessment data with MLflow's built-in scorer names
    const assessmentData: Array<{
      name: string;
      type: 'boolean' | 'numeric' | 'string';
      baseCount: number;
    }> = [
      { name: 'retrieval_relevance', type: 'boolean', baseCount: 14 },
      { name: 'correctness', type: 'boolean', baseCount: 12 },
      { name: 'relevance_to_query', type: 'boolean', baseCount: 10 },
      { name: 'safety', type: 'boolean', baseCount: 8 },
      { name: 'retrieval_groundedness', type: 'boolean', baseCount: 6 },
      { name: 'equivalence', type: 'boolean', baseCount: 5 },
    ];
    
    return assessmentData.map((assessment) => {
      const count = assessment.baseCount;
      
      if (assessment.type === 'boolean') {
        // For boolean assessments, generate avg and split between 0 and 1
        const avg = 0.3 + Math.random() * 0.4; // Random avg between 0.3-0.7
        const count1 = Math.floor(count * avg);
        const count0 = count - count1;
        
        return {
          name: assessment.name,
          count,
          avg: parseFloat(avg.toFixed(2)),
          count0,
          count1,
          type: assessment.type,
        };
      } else if (assessment.type === 'numeric') {
        // For numeric assessments, show avg but no 0/1 counts
        const avg = 0.2 + Math.random() * 0.6; // Random avg between 0.2-0.8
        
        return {
          name: assessment.name,
          count,
          avg: parseFloat(avg.toFixed(2)),
          type: assessment.type,
        };
      } else {
        // For string assessments, no avg or 0/1 counts
        return {
          name: assessment.name,
          count,
          type: assessment.type,
        };
      }
    });
  }, []);

  const totalAssessmentsCount = useMemo(() => {
    return groupedAssessments.reduce((sum, a) => sum + a.count, 0);
  }, [groupedAssessments]);

  return (
    <div
      css={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr 1fr',
        gap: theme.spacing.md,
        '@media (max-width: 1200px)': {
          gridTemplateColumns: '1fr',
        },
      }}
    >
      {/* Grouped Traces Chart */}
      <div
        css={{
          backgroundColor: theme.colors.backgroundPrimary,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
          padding: theme.spacing.md,
        }}
      >
        <div
          css={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            marginBottom: theme.spacing.md,
          }}
        >
          <h3
            css={{
              margin: 0,
              fontSize: theme.typography.fontSizeLg,
              fontWeight: theme.typography.typographyBoldFontWeight,
            }}
          >
            <FormattedMessage
              defaultMessage="Total Traces: {count}"
              description="Title for the grouped traces chart with total count"
              values={{
                count: statistics.totalTraces.toLocaleString(),
              }}
            />
          </h3>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <span
              css={{
                fontSize: theme.typography.fontSizeXs,
                color: theme.colors.textSecondary,
                whiteSpace: 'nowrap',
              }}
            >
              Group by:
            </span>
            <select
              value={traceGroupBy}
              onChange={(e) => setTraceGroupBy(e.target.value as TraceGroupBy)}
              css={{
                fontSize: theme.typography.fontSizeSm,
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.borders.borderRadiusMd,
                backgroundColor: theme.colors.backgroundPrimary,
                color: theme.colors.textPrimary,
                cursor: 'pointer',
                '&:hover': {
                  borderColor: theme.colors.primary,
                },
                '&:focus': {
                  outline: 'none',
                  borderColor: theme.colors.primary,
                },
              }}
            >
              <option value="trace_name">Trace Name</option>
              <option value="trace_status">Trace Status</option>
            </select>
          </div>
        </div>
        
        {isLoading ? (
          <div css={{ minHeight: 300 }}>
            <LegacySkeleton active />
          </div>
        ) : groupedTracesPlotData.length > 0 ? (
          <LazyPlot
            key={`grouped-traces-${traceGroupBy}`}
            data={groupedTracesPlotData}
            layout={groupedTracesLayout}
            config={{
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['sendDataToCloud', 'select2d', 'lasso2d', 'autoScale2d'],
            }}
            css={{ width: '100%' }}
            useResizeHandler
            style={{ width: '100%' }}
          />
        ) : (
          <div
            css={{
              minHeight: 300,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: theme.colors.textSecondary,
            }}
          >
            <FormattedMessage
              defaultMessage="No trace data available"
              description="Empty state for grouped traces chart"
            />
          </div>
        )}
      </div>

      {/* Grouped Tokens Chart */}
      <div
        css={{
          backgroundColor: theme.colors.backgroundPrimary,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
          padding: theme.spacing.md,
        }}
      >
        <div
          css={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            marginBottom: theme.spacing.md,
          }}
        >
          <h3
            css={{
              margin: 0,
              fontSize: theme.typography.fontSizeLg,
              fontWeight: theme.typography.typographyBoldFontWeight,
            }}
          >
            <FormattedMessage
              defaultMessage="Total Tokens: {count}"
              description="Title for the grouped tokens chart with total count"
              values={{
                count: statistics.totalTokens.toLocaleString(),
              }}
            />
          </h3>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <span
              css={{
                fontSize: theme.typography.fontSizeXs,
                color: theme.colors.textSecondary,
                whiteSpace: 'nowrap',
              }}
            >
              Group by:
            </span>
            <select
              value={tokenGroupBy}
              onChange={(e) => setTokenGroupBy(e.target.value as TokenGroupBy)}
              css={{
                fontSize: theme.typography.fontSizeSm,
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.borders.borderRadiusMd,
                backgroundColor: theme.colors.backgroundPrimary,
                color: theme.colors.textPrimary,
                cursor: 'pointer',
                '&:hover': {
                  borderColor: theme.colors.primary,
                },
                '&:focus': {
                  outline: 'none',
                  borderColor: theme.colors.primary,
                },
              }}
            >
              <option value="trace_name">Trace Name</option>
              <option value="token_type">Type</option>
            </select>
          </div>
        </div>
        
        {isLoading ? (
          <div css={{ minHeight: 300 }}>
            <LegacySkeleton active />
          </div>
        ) : groupedTokensPlotData.length > 0 ? (
          <LazyPlot
            key={`grouped-tokens-${tokenGroupBy}`}
            data={groupedTokensPlotData}
            layout={groupedTokensLayout}
            config={{
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['sendDataToCloud', 'select2d', 'lasso2d', 'autoScale2d'],
            }}
            css={{ width: '100%' }}
            useResizeHandler
            style={{ width: '100%' }}
          />
        ) : (
          <div
            css={{
              minHeight: 300,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: theme.colors.textSecondary,
            }}
          >
            <FormattedMessage
              defaultMessage="No token data available"
              description="Empty state for grouped tokens chart"
            />
          </div>
        )}
      </div>

      {/* Grouped Assessments Chart */}
      <div
        css={{
          backgroundColor: theme.colors.backgroundPrimary,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
          padding: theme.spacing.md,
        }}
      >
        <div
          css={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            marginBottom: theme.spacing.md,
          }}
        >
          <div>
            <h3
              css={{
                margin: 0,
                fontSize: theme.typography.fontSizeLg,
                fontWeight: theme.typography.typographyBoldFontWeight,
              }}
            >
              <FormattedMessage
                defaultMessage="Assessments"
                description="Title for the assessments table"
              />
            </h3>
            <div
              css={{
                fontSize: theme.typography.fontSizeSm,
                color: theme.colors.textSecondary,
                marginTop: theme.spacing.xs,
              }}
            >
              {totalAssessmentsCount} Total scores tracked
            </div>
          </div>
        </div>
        
        {isLoading ? (
          <div css={{ minHeight: 200 }}>
            <LegacySkeleton active />
          </div>
        ) : groupedAssessments.length > 0 ? (
          <div css={{ overflowX: 'auto' }}>
            <table
              css={{
                width: '100%',
                borderCollapse: 'collapse',
                fontSize: theme.typography.fontSizeSm,
                marginTop: theme.spacing.md,
              }}
            >
              <thead>
                <tr
                  css={{
                    borderBottom: `1px solid ${theme.colors.border}`,
                  }}
                >
                  <th
                    css={{
                      textAlign: 'left',
                      padding: `${theme.spacing.sm}px ${theme.spacing.xs}px`,
                      fontWeight: theme.typography.typographyBoldFontWeight,
                      color: theme.colors.textPrimary,
                    }}
                  >
                    Name
                  </th>
                  <th
                    css={{
                      textAlign: 'right',
                      padding: `${theme.spacing.sm}px ${theme.spacing.sm}px`,
                      fontWeight: theme.typography.typographyBoldFontWeight,
                      color: theme.colors.textPrimary,
                    }}
                  >
                    #
                  </th>
                  <th
                    css={{
                      textAlign: 'right',
                      padding: `${theme.spacing.sm}px ${theme.spacing.sm}px`,
                      fontWeight: theme.typography.typographyBoldFontWeight,
                      color: theme.colors.textPrimary,
                    }}
                  >
                    Avg
                  </th>
                  <th
                    css={{
                      textAlign: 'right',
                      padding: `${theme.spacing.sm}px ${theme.spacing.sm}px`,
                      fontWeight: theme.typography.typographyBoldFontWeight,
                      color: theme.colors.textPrimary,
                    }}
                  >
                    0
                  </th>
                  <th
                    css={{
                      textAlign: 'right',
                      padding: `${theme.spacing.sm}px ${theme.spacing.sm}px`,
                      fontWeight: theme.typography.typographyBoldFontWeight,
                      color: theme.colors.textPrimary,
                    }}
                  >
                    1
                  </th>
                </tr>
              </thead>
              <tbody>
                {groupedAssessments.map((assessment, index) => (
                  <tr
                    key={assessment.name}
                    css={{
                      borderBottom: index < groupedAssessments.length - 1 ? `1px solid ${theme.colors.grey100}` : 'none',
                      '&:hover': {
                        backgroundColor: theme.colors.backgroundSecondary,
                      },
                    }}
                  >
                    <td
                      css={{
                        padding: `${theme.spacing.sm}px ${theme.spacing.xs}px`,
                        color: theme.colors.textSecondary,
                      }}
                    >
                      {assessment.name}
                    </td>
                    <td
                      css={{
                        textAlign: 'right',
                        padding: `${theme.spacing.sm}px ${theme.spacing.sm}px`,
                        color: theme.colors.textPrimary,
                        fontWeight: theme.typography.typographyBoldFontWeight,
                      }}
                    >
                      {assessment.count}
                    </td>
                    <td
                      css={{
                        textAlign: 'right',
                        padding: `${theme.spacing.sm}px ${theme.spacing.sm}px`,
                        color: theme.colors.textPrimary,
                      }}
                    >
                      {assessment.avg !== undefined ? assessment.avg.toFixed(2) : '-'}
                    </td>
                    <td
                      css={{
                        textAlign: 'right',
                        padding: `${theme.spacing.sm}px ${theme.spacing.sm}px`,
                        color: theme.colors.textPrimary,
                      }}
                    >
                      {assessment.count0 !== undefined ? assessment.count0 : '-'}
                    </td>
                    <td
                      css={{
                        textAlign: 'right',
                        padding: `${theme.spacing.sm}px ${theme.spacing.sm}px`,
                        color: theme.colors.textPrimary,
                      }}
                    >
                      {assessment.count1 !== undefined ? assessment.count1 : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div
            css={{
              minHeight: 200,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: theme.colors.textSecondary,
            }}
          >
            <FormattedMessage
              defaultMessage="No assessment data available"
              description="Empty state for assessments table"
            />
          </div>
        )}
      </div>
    </div>
  );
};

