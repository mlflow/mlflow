export const isPendingApi = (action) => {
  return action.type.endsWith('_PENDING');
};

export const pending = (apiActionType) => {
  return `${apiActionType}_PENDING`;
};

export const isFulfilledApi = (action) => {
  return action.type.endsWith('_FULFILLED');
};

export const fulfilled = (apiActionType) => {
  return `${apiActionType}_FULFILLED`;
};

export const isRejectedApi = (action) => {
  return action.type.endsWith('_REJECTED');
};

export const rejected = (apiActionType) => {
  return `${apiActionType}_REJECTED`;
};

export const getUUID = () => {
  const randomPart = Math.random()
    .toString(36)
    .substring(2, 10);
  return new Date().getTime() + randomPart;
};
