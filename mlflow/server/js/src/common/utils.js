import ErrorCodes from '../sdk/ErrorCodes';

export const shouldRender404 = (requests, requestIdsToCheck) => {
  const requestsToCheck = requests.filter((request) =>
    requestIdsToCheck.includes(request.id),
  );
  return requestsToCheck.some((request) => {
    const { error } = request;
    return error && error.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST;
  });
};

// TODO(Zangr) move to /experiment-tracking after folder refactor
export const ColumnTypes = {
  ATTRIBUTES: 'attributes',
  PARAMS: 'params',
  METRICS: 'metrics',
  TAGS: 'tags',
};
