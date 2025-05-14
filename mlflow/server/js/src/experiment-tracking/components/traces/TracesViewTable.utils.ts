import { ColumnDef } from '@tanstack/react-table';
import { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import { Interpolation, Theme } from '@emotion/react';

export type TracesColumnDef = ColumnDef<ModelTraceInfoWithRunName> & {
  meta?: {
    styles?: Interpolation<Theme>;
    multiline?: boolean;
  };
};

export const getHeaderSizeClassName = (id: string) => `--header-${id}-size`;
export const getColumnSizeClassName = (id: string) => `--col-${id}-size`;
