import type { ColumnDef } from '@tanstack/react-table';
import type { Endpoint } from '../types';
import type { Interpolation, Theme } from '@emotion/react';
import { type MessageDescriptor, defineMessage } from '@databricks/i18n';

export type RoutesColumnDef = ColumnDef<Endpoint> & {
  meta?: {
    styles?: Interpolation<Theme>;
  };
};

export const getHeaderSizeClassName = (id: string) => `--header-${id}-size`;
export const getColumnSizeClassName = (id: string) => `--col-${id}-size`;

export enum RoutesTableColumns {
  name = 'name',
  description = 'description',
  secretName = 'secret_name',
  modelName = 'model_name',
  provider = 'provider',
  tags = 'tags',
  bindingCount = 'binding_count',
  createdAt = 'created_at',
  lastUpdatedAt = 'last_updated_at',
  createdBy = 'created_by',
  lastUpdatedBy = 'last_updated_by',
}

export const RoutesTableColumnLabels: Record<RoutesTableColumns, MessageDescriptor> = {
  [RoutesTableColumns.name]: defineMessage({
    defaultMessage: 'Name',
    description: 'Routes table > name column header',
  }),
  [RoutesTableColumns.description]: defineMessage({
    defaultMessage: 'Description',
    description: 'Routes table > description column header',
  }),
  [RoutesTableColumns.secretName]: defineMessage({
    defaultMessage: 'Secret / API Key',
    description: 'Routes table > secret name column header',
  }),
  [RoutesTableColumns.modelName]: defineMessage({
    defaultMessage: 'Model',
    description: 'Routes table > model name column header',
  }),
  [RoutesTableColumns.provider]: defineMessage({
    defaultMessage: 'Provider',
    description: 'Routes table > provider column header',
  }),
  [RoutesTableColumns.tags]: defineMessage({
    defaultMessage: 'Tags',
    description: 'Routes table > tags column header',
  }),
  [RoutesTableColumns.bindingCount]: defineMessage({
    defaultMessage: 'Bindings',
    description: 'Routes table > binding count column header',
  }),
  [RoutesTableColumns.createdAt]: defineMessage({
    defaultMessage: 'Created',
    description: 'Routes table > created at column header',
  }),
  [RoutesTableColumns.lastUpdatedAt]: defineMessage({
    defaultMessage: 'Last Updated',
    description: 'Routes table > last updated at column header',
  }),
  [RoutesTableColumns.createdBy]: defineMessage({
    defaultMessage: 'Created By',
    description: 'Routes table > created by column header',
  }),
  [RoutesTableColumns.lastUpdatedBy]: defineMessage({
    defaultMessage: 'Last Updated By',
    description: 'Routes table > last updated by column header',
  }),
};
