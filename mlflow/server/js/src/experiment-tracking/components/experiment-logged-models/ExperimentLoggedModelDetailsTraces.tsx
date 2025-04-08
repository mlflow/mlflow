import { useEffect, useMemo } from 'react';
import type { LoggedModelProto } from '../../types';
import { TracesView } from '../traces/TracesView';

export const ExperimentLoggedModelDetailsTraces = ({ loggedModel }: { loggedModel: LoggedModelProto }) => {
  const experimentIds = useMemo(() => [loggedModel.info?.experiment_id ?? ''], [loggedModel.info?.experiment_id]);

  if (!loggedModel.info?.experiment_id) {
    return null;
  }
  return (
    <div css={{ height: '100%', overflow: 'hidden' }}>
      <TracesView experimentIds={experimentIds} loggedModelId={loggedModel.info?.model_id} />
    </div>
  );
};
