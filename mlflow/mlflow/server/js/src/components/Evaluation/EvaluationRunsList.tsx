import React from 'react';
import { EvaluationMenu } from './EvaluationMenu';

export const EvaluationRunsList = ({ runUuid }: { runUuid: string }) => {
  return (
    <div>
      <h3>Evaluations for Run: {runUuid}</h3>
      <EvaluationMenu runUuid={runUuid} />
    </div>
  );
};

export default EvaluationRunsList;
