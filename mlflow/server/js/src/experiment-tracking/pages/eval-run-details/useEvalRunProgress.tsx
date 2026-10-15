import { useMemo } from 'react';
import { useGetRunQuery } from '../../components/run-page/hooks/useGetRunQuery';
import { MlflowRunStatus } from '../../../graphql/__generated__/graphql';
import type { KeyValueEntity } from '../../../common/types';

const TERMINAL_STATUSES = [MlflowRunStatus.FINISHED, MlflowRunStatus.FAILED, MlflowRunStatus.KILLED];

/** Loads the run behind the evaluation run details page: its name, status, experiment and tags. */
export const useEvalRunProgress = ({ runUuid }: { runUuid: string }) => {
  const { data: run, loading } = useGetRunQuery({ runUuid });
  const status = run?.info?.status ?? undefined;
  const runName = run?.info?.runName ?? undefined;
  const experiment = run?.experiment ?? undefined;
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

  return {
    isLoading: loading && !run,
    status,
    runName,
    experiment,
    runTags,
    isFinished,
  };
};
