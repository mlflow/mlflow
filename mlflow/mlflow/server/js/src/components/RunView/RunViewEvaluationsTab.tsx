import React from 'react';
import { EvaluationRunsList } from '../Evaluation/EvaluationRunsList';

export const RunViewEvaluationsTab = ({ runUuid }: { runUuid: string }) => {
  return (
    <div style={{ padding: 16 }}>
      <EvaluationRunsList runUuid={runUuid} />
    </div>
  );
};

export default RunViewEvaluationsTab;
