export const TAG_COLUMN_PREFIX = 'tag:';
export const METADATA_COLUMN_PREFIX = 'custom_metadata:';

export const isCustomTraceColumnId = (id: string) =>
  id.startsWith(TAG_COLUMN_PREFIX) || id.startsWith(METADATA_COLUMN_PREFIX);
