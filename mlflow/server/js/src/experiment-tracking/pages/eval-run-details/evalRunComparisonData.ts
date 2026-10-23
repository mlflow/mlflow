export const normalizeComparisonRunUuids = (baselineRunUuid: string, compareRunUuids: string[]): string[] => [
  ...new Set([baselineRunUuid, ...compareRunUuids].filter(Boolean)),
];
