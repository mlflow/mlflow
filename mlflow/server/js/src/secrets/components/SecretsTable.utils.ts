import type { ColumnDef } from '@tanstack/react-table';
import type { Secret } from '../types';
import type { Interpolation, Theme } from '@emotion/react';
import { type MessageDescriptor, defineMessage } from '@databricks/i18n';

export type SecretsColumnDef = ColumnDef<Secret> & {
  meta?: {
    styles?: Interpolation<Theme>;
  };
};

export const getHeaderSizeClassName = (id: string) => `--header-${id}-size`;
export const getColumnSizeClassName = (id: string) => `--col-${id}-size`;

export enum SecretsTableColumns {
  secretName = 'secret_name',
  maskedValue = 'masked_value',
  isShared = 'is_shared',
  owner = 'owner',
  createdAt = 'created_at',
  updatedAt = 'updated_at',
  bindingCount = 'binding_count',
}

export const SecretsTableColumnLabels: Record<SecretsTableColumns, MessageDescriptor> = {
  [SecretsTableColumns.secretName]: defineMessage({
    defaultMessage: 'Name',
    description: 'Secrets table > secret name column header',
  }),
  [SecretsTableColumns.maskedValue]: defineMessage({
    defaultMessage: 'Masked Value',
    description: 'Secrets table > masked value column header',
  }),
  [SecretsTableColumns.isShared]: defineMessage({
    defaultMessage: 'Shared',
    description: 'Secrets table > is shared column header',
  }),
  [SecretsTableColumns.owner]: defineMessage({
    defaultMessage: 'Owner',
    description: 'Secrets table > owner column header',
  }),
  [SecretsTableColumns.createdAt]: defineMessage({
    defaultMessage: 'Created',
    description: 'Secrets table > created at column header',
  }),
  [SecretsTableColumns.updatedAt]: defineMessage({
    defaultMessage: 'Updated',
    description: 'Secrets table > updated at column header',
  }),
  [SecretsTableColumns.bindingCount]: defineMessage({
    defaultMessage: 'Bindings',
    description: 'Secrets table > binding count column header',
  }),
};
