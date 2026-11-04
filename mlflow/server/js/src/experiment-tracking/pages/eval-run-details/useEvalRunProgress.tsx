import { useMemo } from 'react';
import { useGetRunQuery } from '../../components/run-page/hooks/useGetRunQuery';
import { MlflowRunStatus } from '../../../graphql/__generated__/graphql';
import type { KeyValueEntity } from '../../../common/types';
import { ErrorCodes } from '../../../common/constants';
import { getGraphQLErrorMessage } from '../../../graphql/get-graphql-error';

const TERMINAL_STATUSES = [MlflowRunStatus.FINISHED, MlflowRunStatus.FAILED, MlflowRunStatus.KILLED];
const EMPTY_METRICS: { key?: string | null; value?: number | null }[] = [];

/** Loads the run behind the evaluation run details page: its name, status, experiment, metrics and tags. */
export const useEvalRunProgress = ({ runUuid }: { runUuid: string }) => {
  const { data: run, loading, apiError, apolloError } = useGetRunQuery({ runUuid });
  const status = run?.info?.status ?? undefined;
  const runName = run?.info?.runName ?? undefined;
  const experiment = run?.experiment ?? undefined;
  const runMetrics = run?.data?.metrics ?? EMPTY_METRICS;
  const runTags = useMemo<Record<string, KeyValueEntity>>(
    () =>
      Object.fromEntries(
        (run?.data?.tags ?? [])
          .filter((tag): tag is typeof tag & { key: string } => Boolean(tag.key))
          .map((tag) => [tag.key, { key: tag.key, value: tag.value ?? '' }]),
      ),
    [run?.data?.tags],
  );
  const isFinished = status ? TERMINAL_STATUSES.includes(status) : false;
  // Same checks as the run page: the API error code, or the GraphQL message for a missing run.
  const isNotFound =
    apiError?.code === ErrorCodes.RESOURCE_DOES_NOT_EXIST ||
    Boolean(apolloError && getGraphQLErrorMessage(apolloError).match(/not found$/));
  // A nested resolver can fail while the run itself loads, so only report an error without run data.
  const loadError =
    !run && (apiError || apolloError) ? (getGraphQLErrorMessage(apiError ?? apolloError) ?? '') : undefined;

  return {
    isLoading: loading && !run,
    isNotFound,
    loadError,
    status,
    runName,
    experiment,
    runMetrics,
    runTags,
    isFinished,
  };
};
