import { fulfilled, getUUID, isFulfilledApi, isPendingApi, isRejectedApi, pending, rejected } from './ActionUtils';

describe('ActionUtils', () => {
  it('getUUID', () => {
    const uuid = getUUID();
    expect(uuid.length).toEqual(new Date().getTime().toString().length + 8);
  });

  it('apiActionTypes', () => {
    const actionType = 'GRAB_ME_A_COKE';
    [pending(actionType), fulfilled(actionType), rejected(actionType)].forEach((type) => {
      expect(isPendingApi({ type })).toEqual(type === pending(actionType));
      expect(isFulfilledApi({ type })).toEqual(type === fulfilled(actionType));
      expect(isRejectedApi({ type })).toEqual(type === rejected(actionType));
    });
  });
});
