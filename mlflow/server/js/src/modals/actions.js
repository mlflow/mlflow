import Immutable from 'immutable';

export const SHOW_MODAL = 'SHOW_MODAL';
export const HIDE_MODAL = 'HIDE_MODAL';


export function showModal(modalName, modalParams) {
  return {
    type: SHOW_MODAL,
    payload: Immutable.fromJS({
      modalName,
      modalParams
    }),
  }
}

export function hideModal() {
  return {
    type: HIDE_MODAL,
  }
}
