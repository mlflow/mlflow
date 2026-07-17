export const formatTokens = (tokens: number | null | undefined): string | null => {
  if (tokens === null || tokens === undefined) return null;
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(0)}K`;
  return tokens.toString();
};

export const formatCost = (cost: number | null | undefined): string | null => {
  if (cost === null || cost === undefined) return null;
  if (cost === 0) return 'Free';
  const perMillion = cost * 1_000_000;
  if (perMillion < 0.01) return `$${perMillion.toFixed(4)}/1M`;
  return `$${perMillion.toFixed(2)}/1M`;
};

export const extractModelDate = (modelName: string): Date | null => {
  const yyyymmddMatch = modelName.match(/(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)/);
  if (yyyymmddMatch) {
    const [, year, month, day] = yyyymmddMatch;
    return new Date(parseInt(year, 10), parseInt(month, 10) - 1, parseInt(day, 10));
  }

  const isoMatch = modelName.match(/(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])/);
  if (isoMatch) {
    const [, year, month, day] = isoMatch;
    return new Date(parseInt(year, 10), parseInt(month, 10) - 1, parseInt(day, 10));
  }

  const mmddMatch = modelName.match(/-(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?:-|$)/);
  if (mmddMatch) {
    const [, month, day] = mmddMatch;
    const currentYear = new Date().getFullYear();
    return new Date(currentYear, parseInt(month, 10) - 1, parseInt(day, 10));
  }

  return null;
};

export const extractModelVersion = (modelName: string): number => {
  const nameWithoutDates = modelName.replace(/-\d{4}-\d{2}-\d{2}$/, '').replace(/-\d{8}$/, '');

  const oSeriesMatch = nameWithoutDates.match(/(?:^|[/-])o(\d+)(?:-|$)/);
  if (oSeriesMatch) {
    return parseInt(oSeriesMatch[1], 10);
  }

  const dotVersion = nameWithoutDates.match(/-(\d+)\.(\d+)/);
  if (dotVersion) {
    return parseFloat(`${dotVersion[1]}.${dotVersion[2]}`);
  }

  const compactVersion = nameWithoutDates.match(/-(\d)(\d)-/);
  if (compactVersion) {
    return parseFloat(`${compactVersion[1]}.${compactVersion[2]}`);
  }

  const dashVersionEnd = nameWithoutDates.match(/(\d+)-(\d{1,2})$/);
  if (dashVersionEnd) {
    return parseFloat(`${dashVersionEnd[1]}.${dashVersionEnd[2]}`);
  }

  const dashVersionMid = nameWithoutDates.match(/(\d+)-(\d{1,2})-[a-z]/i);
  if (dashVersionMid) {
    return parseFloat(`${dashVersionMid[1]}.${dashVersionMid[2]}`);
  }

  const singleVersion = nameWithoutDates.match(/-(\d)o?(?:-|$)/);
  if (singleVersion) {
    return parseInt(singleVersion[1], 10);
  }

  return 0;
};

export const sortModelsByDate = <T extends { model: string }>(models: T[]): T[] => {
  return [...models].sort((a, b) => {
    const versionA = extractModelVersion(a.model);
    const versionB = extractModelVersion(b.model);

    if (versionA !== versionB) {
      return versionB - versionA;
    }

    const dateA = extractModelDate(a.model);
    const dateB = extractModelDate(b.model);

    if (dateA && dateB) {
      return dateB.getTime() - dateA.getTime();
    }

    if (dateA && !dateB) return -1;
    if (!dateA && dateB) return 1;

    return b.model.localeCompare(a.model);
  });
};
