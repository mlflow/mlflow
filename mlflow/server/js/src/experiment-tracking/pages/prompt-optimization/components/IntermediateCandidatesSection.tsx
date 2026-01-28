import { useState, useMemo } from 'react';
import {
  ChevronDownIcon,
  ChevronRightIcon,
  Spacer,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { QueryFunctionContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../../sdk/MlflowService';
import { getArtifactContent, getArtifactLocationUrl } from '../../../../common/utils/ArtifactUtils';

interface IntermediateCandidatesSectionProps {
  runId: string;
}

interface CandidateScores {
  iteration: number;
  scores: Record<string, number>;
  aggregateScore?: number;
}

interface ArtifactFile {
  path: string;
  is_dir?: boolean;
}

interface ArtifactsListResponse {
  files?: ArtifactFile[];
}

type ArtifactsQueryKey = ['optimization_candidates_artifacts', string];
type ScoresQueryKey = ['optimization_candidates_scores', string, number[]];

const artifactsQueryFn = ({ queryKey }: QueryFunctionContext<ArtifactsQueryKey>) => {
  const [, runId] = queryKey;
  return MlflowService.listArtifacts({ run_uuid: runId, path: 'prompt_candidates' }) as Promise<ArtifactsListResponse>;
};

export const IntermediateCandidatesSection = ({ runId }: IntermediateCandidatesSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const [expandedIterations, setExpandedIterations] = useState<Set<number>>(new Set());
  const [loadedPrompts, setLoadedPrompts] = useState<Record<number, string>>({});
  const [loadingPrompts, setLoadingPrompts] = useState<Set<number>>(new Set());

  // Fetch the list of iteration folders from artifacts
  const { data: artifactsData, isLoading: artifactsLoading } = useQuery<
    ArtifactsListResponse,
    Error,
    ArtifactsListResponse,
    ArtifactsQueryKey
  >(['optimization_candidates_artifacts', runId], {
    queryFn: artifactsQueryFn,
    enabled: !!runId,
    retry: false,
  });

  // Parse iteration folders from artifacts
  const iterations = useMemo(() => {
    if (!artifactsData?.files) return [] as number[];

    const iterationFolders = artifactsData.files
      .filter((file: ArtifactFile) => file.is_dir && file.path.startsWith('prompt_candidates/iteration_'))
      .map((file: ArtifactFile) => {
        const match = file.path.match(/iteration_(\d+)/);
        return match ? parseInt(match[1], 10) : -1;
      })
      .filter((num: number) => num >= 0)
      .sort((a: number, b: number) => a - b);

    return iterationFolders;
  }, [artifactsData]);

  const scoresQueryFn = async ({ queryKey }: QueryFunctionContext<ScoresQueryKey>) => {
    const [, runUuid, iterationList] = queryKey;
    const scoresMap: Record<number, CandidateScores> = {};

    await Promise.all(
      iterationList.map(async (iteration: number) => {
        try {
          const scoresPath = `prompt_candidates/iteration_${iteration}/scores.json`;
          const artifactUrl = getArtifactLocationUrl(scoresPath, runUuid);
          const scoresJson = await getArtifactContent<string>(artifactUrl);
          const scores = JSON.parse(scoresJson);
          scoresMap[iteration] = {
            iteration,
            scores,
            aggregateScore: scores['aggregate'] ?? scores['Aggregate'],
          };
        } catch (error) {
          // Score file might not exist for all iterations
          console.warn(`Failed to load scores for iteration ${iteration}:`, error);
        }
      }),
    );

    return scoresMap;
  };

  // Fetch scores for all iterations
  const { data: scoresData } = useQuery<
    Record<number, CandidateScores>,
    Error,
    Record<number, CandidateScores>,
    ScoresQueryKey
  >(['optimization_candidates_scores', runId, iterations], {
    queryFn: scoresQueryFn,
    enabled: !!runId && iterations.length > 0,
    retry: false,
  });

  const toggleIteration = async (iteration: number) => {
    const newExpanded = new Set(expandedIterations);

    if (newExpanded.has(iteration)) {
      newExpanded.delete(iteration);
    } else {
      newExpanded.add(iteration);

      // Load prompt content if not already loaded
      if (!loadedPrompts[iteration] && !loadingPrompts.has(iteration)) {
        setLoadingPrompts((prev) => new Set(prev).add(iteration));

        try {
          // Try to find the prompt file in this iteration
          const iterationArtifacts = (await MlflowService.listArtifacts({
            run_uuid: runId,
            path: `prompt_candidates/iteration_${iteration}`,
          })) as ArtifactsListResponse;

          const promptFile = iterationArtifacts.files?.find(
            (f: ArtifactFile) => f.path.endsWith('.txt') || f.path.endsWith('.md'),
          );

          if (promptFile) {
            const artifactUrl = getArtifactLocationUrl(promptFile.path, runId);
            const promptContent = await getArtifactContent<string>(artifactUrl);
            setLoadedPrompts((prev) => ({ ...prev, [iteration]: promptContent }));
          }
        } catch (error) {
          console.error(`Failed to load prompt for iteration ${iteration}:`, error);
        } finally {
          setLoadingPrompts((prev) => {
            const next = new Set(prev);
            next.delete(iteration);
            return next;
          });
        }
      }
    }

    setExpandedIterations(newExpanded);
  };

  if (artifactsLoading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Spinner size="small" />
        <Typography.Text>
          <FormattedMessage defaultMessage="Loading candidates..." description="Loading candidates message" />
        </Typography.Text>
      </div>
    );
  }

  if (iterations.length === 0) {
    return (
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="No intermediate candidates available yet."
          description="No candidates message"
        />
      </Typography.Text>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {iterations.map((iteration: number) => {
        const isExpanded = expandedIterations.has(iteration);
        const isLoading = loadingPrompts.has(iteration);
        const candidateScores = scoresData?.[iteration];
        const promptContent = loadedPrompts[iteration];

        return (
          <div
            key={iteration}
            css={{
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.general.borderRadiusBase,
              overflow: 'hidden',
            }}
          >
            {/* Iteration Header */}
            <button
              type="button"
              onClick={() => toggleIteration(iteration)}
              css={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                padding: theme.spacing.sm,
                background: theme.colors.backgroundSecondary,
                border: 'none',
                cursor: 'pointer',
                textAlign: 'left',
                '&:hover': {
                  background: theme.colors.actionTertiaryBackgroundHover,
                },
              }}
            >
              {isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="Iteration {iteration}"
                  description="Iteration header"
                  values={{ iteration }}
                />
              </Typography.Text>
              {candidateScores?.aggregateScore !== undefined && (
                <Typography.Text color="secondary" css={{ marginLeft: 'auto' }}>
                  <FormattedMessage
                    defaultMessage="Score: {score}"
                    description="Aggregate score display"
                    values={{ score: candidateScores.aggregateScore.toFixed(3) }}
                  />
                </Typography.Text>
              )}
            </button>

            {/* Expanded Content */}
            {isExpanded && (
              <div css={{ padding: theme.spacing.md }}>
                {/* Scores Grid */}
                {candidateScores && Object.keys(candidateScores.scores).length > 0 && (
                  <>
                    <Typography.Text bold size="sm">
                      <FormattedMessage defaultMessage="Scores" description="Scores label" />
                    </Typography.Text>
                    <div
                      css={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
                        gap: theme.spacing.sm,
                        marginTop: theme.spacing.xs,
                        marginBottom: theme.spacing.md,
                      }}
                    >
                      {Object.entries(candidateScores.scores).map(([key, value]) => (
                        <div
                          key={key}
                          css={{
                            padding: theme.spacing.xs,
                            background: theme.colors.backgroundSecondary,
                            borderRadius: theme.general.borderRadiusBase,
                          }}
                        >
                          <Typography.Text color="secondary" size="sm">
                            {key}
                          </Typography.Text>
                          <Typography.Text>
                            {typeof value === 'number' ? value.toFixed(3) : String(value)}
                          </Typography.Text>
                        </div>
                      ))}
                    </div>
                  </>
                )}

                {/* Prompt Content */}
                <Typography.Text bold size="sm">
                  <FormattedMessage defaultMessage="Prompt" description="Prompt label" />
                </Typography.Text>
                <Spacer size="xs" />
                {isLoading ? (
                  <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                    <Spinner size="small" />
                    <Typography.Text color="secondary">
                      <FormattedMessage defaultMessage="Loading prompt..." description="Loading prompt message" />
                    </Typography.Text>
                  </div>
                ) : promptContent ? (
                  <pre
                    css={{
                      background: theme.colors.backgroundSecondary,
                      padding: theme.spacing.md,
                      borderRadius: theme.general.borderRadiusBase,
                      overflow: 'auto',
                      maxHeight: 400,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      fontSize: theme.typography.fontSizeSm,
                      fontFamily: 'monospace',
                      margin: 0,
                    }}
                  >
                    {promptContent}
                  </pre>
                ) : (
                  <Typography.Text color="secondary">
                    <FormattedMessage
                      defaultMessage="Prompt content not available"
                      description="No prompt content message"
                    />
                  </Typography.Text>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
