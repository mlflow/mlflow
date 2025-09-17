import { ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import type { RunsChartsRunData } from '../components/RunsCharts.common';

export const useImageSliderStepMarks = ({
  data,
  selectedImageKeys,
}: {
  data: RunsChartsRunData[];
  selectedImageKeys: string[];
}) => {
  const stepMarks = data.reduce((acc, run: RunsChartsRunData) => {
    for (const imageKey of Object.keys(run.images)) {
      if (selectedImageKeys?.includes(imageKey)) {
        const metadata = run.images[imageKey];
        for (const meta of Object.values(metadata)) {
          if (meta.step !== undefined) {
            acc[meta.step] = {
              style: { display: 'none' },
              label: '',
            };
          }
        }
      }
    }
    return acc;
  }, {} as Record<number, any>);

  return {
    stepMarks,
    maxMark: Math.max(...Object.keys(stepMarks).map(Number)),
    minMark: Math.min(...Object.keys(stepMarks).map(Number)),
  };
};
