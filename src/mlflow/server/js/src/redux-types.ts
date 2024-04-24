import type { ExperimentStoreEntities } from './experiment-tracking/types';
import type { ModelGatewayReduxState } from './experiment-tracking/reducers/ModelGatewayReducer';
import type { EvaluationDataReduxState } from './experiment-tracking/reducers/EvaluationDataReducer';
import type {
  ApisReducerReduxState,
  ComparedExperimentsReducerReduxState,
  ViewsReducerReduxState,
} from './experiment-tracking/reducers/Reducers';

/**
 * Shape of redux state defined by the combined root reducer
 */
export type ReduxState = {
  entities: ExperimentStoreEntities;
  apis: ApisReducerReduxState;
  views: ViewsReducerReduxState;
  modelGateway: ModelGatewayReduxState;
  evaluationData: EvaluationDataReduxState;
  comparedExperiments: ComparedExperimentsReducerReduxState;
};

// Redux type definitions combining redux-thunk & redux-promise-middleware types
// https://gist.github.com/apieceofbart/8b5ab61f1bed29ef25f3b135818e5448

export type Fulfilled<T extends string> = `${T}_FULFILLED`;
export type Pending<T extends string> = `${T}_PENDING`;
export type Rejected<T extends string> = `${T}_REJECTED`;

type AsyncFunction<R = any> = () => Promise<R>;
type AsyncPayload<R = any> =
  | Promise<R>
  | AsyncFunction<R>
  | {
      promise: Promise<R> | AsyncFunction<R>;
      data?: any;
    };

export interface AsyncAction<R = any, M = any> {
  type: string;
  payload: AsyncPayload<R>;
  meta?: M;
  error?: boolean;
}

type AsyncActionResult<A> = A extends AsyncAction<infer R> ? R : never;

/**
 * Type denoting an action after transforming into result fulfilled shape
 */
export type AsyncFulfilledAction<A extends AsyncAction, Type extends string = Fulfilled<A['type']>> = Omit<
  A,
  'type' | 'payload'
> & {
  type: Type;
  payload: AsyncActionResult<A>;
};

export type AsyncPendingAction<A extends AsyncAction, Type extends string = Pending<A['type']>> = Omit<
  A,
  'type' | 'payload'
> & {
  type: Type;
};

export type AsyncRejectedAction<A extends AsyncAction, Type extends string = Rejected<A['type']>> = Omit<
  A,
  'type' | 'payload'
> & {
  type: Type;
  payload: Error;
};

type FulfilledDispatchResult<A extends AsyncAction> = {
  action: AsyncFulfilledAction<A>;
  value: AsyncActionResult<A>;
};

/**
 * Result type of the async, thunked dispatch
 */
export type AsyncDispatchReturns<T> = T extends AsyncAction ? Promise<FulfilledDispatchResult<T>> : T;

export type ThunkDispatchReturns<S, E, A> = A extends ThunkAction<infer R, S, E> ? R : A;

/**
 * Type of dispatch() compatible with the promise-middleware and redux-thunk
 */
export interface ThunkDispatch<S = ReduxState, E = any> {
  <A>(action: A): AsyncDispatchReturns<ThunkDispatchReturns<S, E, A>>;
}

/**
 * Type of thunked action compatible with the promise-middleware and redux-thunk
 */
export type ThunkAction<R, S, E = null> = (dispatch: ThunkDispatch<S, E>, getState: () => S, extraArgument: E) => R;
