import { useState } from 'react';

export const useEvaluationRunFilter = () => {
  const [mode, setMode] = useState<'show_all' | 'hide_finished' | 'show_first_10'>('show_all');

  const applyFilter = (runs: any[]) => {
    if (mode === 'show_all') return runs;

    if (mode === 'hide_finished') {
      return runs.filter((r) => r.status !== 'FINISHED');
    }

    if (mode === 'show_first_10') {
      return runs.slice(0, 10);
    }

    return runs;
  };

  return { mode, setMode, applyFilter };
};
