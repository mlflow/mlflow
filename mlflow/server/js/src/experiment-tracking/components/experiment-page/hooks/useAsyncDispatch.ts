import { useDispatch } from 'react-redux';
import type { AnyAction } from 'redux';

/**
 * Since we're using redux middlewares that promisify action dispatch
 * ('redux-promise-middleware'), we use this method as a sugar for useDispatch()
 * with correct typings.
 */
export const useAsyncDispatch = useDispatch as () => (action: AnyAction) => Promise<AnyAction>;
