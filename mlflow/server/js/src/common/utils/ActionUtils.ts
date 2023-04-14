export const isPendingApi = (action: any) => {
  return action.type.endsWith('_PENDING');
};

export const pending = (apiActionType: any) => {
  return `${apiActionType}_PENDING`;
};

export const isFulfilledApi = (action: any) => {
  return action.type.endsWith('_FULFILLED');
};

export const fulfilled = (apiActionType: any) => {
  return `${apiActionType}_FULFILLED`;
};

export const isRejectedApi = (action: any) => {
  return action.type.endsWith('_REJECTED');
};

export const rejected = (apiActionType: any) => {
  return `${apiActionType}_REJECTED`;
};

export const getUUID = () => {
  const randomPart = Math.random().toString(36).substring(2, 10);
  return new Date().getTime() + randomPart;
};
