import React from 'react';
import { EXPERIMENT_ARTIFACT_LOCATION_CHAR_LIMIT } from '@mlflow/mlflow/src/experiment-tracking/constants';

export interface ExperimentViewArtifactLocationProps {
  artifactLocation: string;
}

export const ExperimentViewArtifactLocation = ({ artifactLocation }: ExperimentViewArtifactLocationProps) => {
  return (
    <span
      css={{
        maxWidth: 400,
        overflow: 'hidden',
        whiteSpace: 'nowrap',
        textOverflow: 'ellipsis',
      }}
    >
      {artifactLocation}
    </span>
  );
};
