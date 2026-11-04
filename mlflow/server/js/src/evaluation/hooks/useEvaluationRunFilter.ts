import { useMemo } from 'react';

export const useEvaluationRunFilter = (runs: any[], mode: string) => {
  return useMemo(() => {
    switch (mode) {
      case 'hide_finished':
        return runs.filter((run) => run.status !== 'FINISHED');
      case 'show_first_10':
        return runs.slice(0, 10);
      case 'show_all':
      default:
        return runs;
    }
  }, [runs, mode]);
};
