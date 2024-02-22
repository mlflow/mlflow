import React from 'react';

export interface ExperimentViewArtifactLocationProps {
  artifactLocation: string;
}

export const ExperimentViewArtifactLocation = ({ artifactLocation }: ExperimentViewArtifactLocationProps) => {
  return <>{artifactLocation}</>;
};
