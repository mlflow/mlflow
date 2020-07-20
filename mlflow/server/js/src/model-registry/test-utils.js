export const mockRegisteredModelDetailed = (name, latestVersions = []) => {
  return {
    creation_timestamp: '1571344731467',
    last_updated_timestamp: '1573581360069',
    latest_versions: latestVersions,
    name,
  };
};

export const mockModelVersionDetailed = (name, version, stage, status) => {
  return {
    name,
    // Use version-based timestamp to make creation_timestamp differ across model versions
    // and prevent React duplicate key warning.
    creation_timestamp: version.toString(),
    last_updated_timestamp: (version + 1).toString(),
    user_id: 'richard@example.com',
    current_stage: stage,
    description: '',
    source: 'path/to/model',
    run_id: 'b99a0fc567ae4d32994392c800c0b6ce',
    status,
    version,
  };
};

export const mockGetFieldValue = (comment, archive) => {
  return (key) => {
    if (key === 'comment') {
      return comment;
    } else if (key === 'archiveExistingVersions') {
      return archive;
    }
    throw new Error('Missing mockGetFieldValue key');
  };
};
