import { useCallback, useReducer } from 'react';
function treeGridReducer(state, action) {
    switch (action.type) {
        case 'TOGGLE_ROW_EXPANDED':
            return {
                ...state,
                expandedRows: { ...state.expandedRows, [action.rowId]: !state.expandedRows[action.rowId] },
            };
        case 'SET_ACTIVE_ROW_ID':
            return { ...state, activeRowId: action.rowId };
        default:
            return state;
    }
}
export function useDefaultTreeGridState({ initialState = { expandedRows: {} }, }) {
    const [state, dispatch] = useReducer(treeGridReducer, { ...initialState, activeRowId: null });
    const toggleRowExpanded = useCallback((rowId) => {
        dispatch({ type: 'TOGGLE_ROW_EXPANDED', rowId });
    }, []);
    const setActiveRowId = useCallback((rowId) => {
        dispatch({ type: 'SET_ACTIVE_ROW_ID', rowId });
    }, []);
    return {
        ...state,
        toggleRowExpanded,
        setActiveRowId,
    };
}
//# sourceMappingURL=useDefaultTreeGridState.js.map