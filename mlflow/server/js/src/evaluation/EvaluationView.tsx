import React, { useEffect, useState } from 'react';
import { Spinner } from '../../common/components/Spinner';
import { getEvaluationRuns } from '../services/getEvaluationRuns';
import { EvaluationRunVisibilityMenu } from './components/EvaluationRunVisibilityMenu';
import { useEvaluationRunFilter } from './hooks/useEvaluationRunFilter';

export const EvaluationView = () => {
  const [runs, setRuns] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  // Hook para filtrado
  const { mode, setMode, applyFilter } = useEvaluationRunFilter();

  useEffect(() => {
    const load = async () => {
      const result = await getEvaluationRuns();
      setRuns(result);
      setLoading(false);
    };
    load();
  }, []);

  if (loading) {
    return <Spinner />;
  }

  // Aplicar filtro
  const filteredRuns = applyFilter(runs);

  return (
    <div style={{ padding: '20px' }}>
      <h2>Evaluation Results</h2>

      {/* Menú de visibilidad */}
      <div style={{ marginBottom: '20px' }}>
        <EvaluationRunVisibilityMenu onChangeVisibility={setMode} />
      </div>

      {/* Lista simple de runs para demostrar el filtro */}
      <div>
        <h3>Runs displayed ({filteredRuns.length})</h3>
        <ul>
          {filteredRuns.map((run: any, index: number) => (
            <li key={index}>
              {run.name} — Status: {run.status}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
