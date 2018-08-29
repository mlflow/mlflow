import Immutable from 'immutable';

import { SHOW_MODAL, HIDE_MODAL} from './actions';

const initialState = Immutable.fromJS({
  currentModal: {modalName: undefined, modalParams: {}},
  previousModal: {modalName: undefined, modalParams: {}},
});

export function getCurrentModal(state) {
  return state.modals.get('currentModal');
}

export function getPreviousModal(state) {
  return state.modals.get('previousModal');
}

export default function modalsReducer(state = initialState, action) {
  switch (action.type) {
    case SHOW_MODAL: {
      return state
        .set('previousModal', state.get('currentModal'))
        .set('currentModal', action.payload);

    }
    case HIDE_MODAL: {
      return state
        .set('previousModal', state.get('currentModal'))
        .set('currentModal', initialState.get('currentModal'));
    }
    default:
      return state;
  }
}
