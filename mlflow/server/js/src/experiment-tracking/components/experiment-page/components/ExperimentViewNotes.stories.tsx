import React from 'react';
import { IntlProvider } from 'react-intl';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { applyMiddleware, compose, createStore } from 'redux';
import promiseMiddleware from 'redux-promise-middleware';
import { ExperimentTag } from '../../../sdk/MlflowMessages';
import type { ExperimentEntity } from '../../../types';
import { GetExperimentsContextProvider } from '../contexts/GetExperimentsContext';
import { ExperimentViewNotes } from './ExperimentViewNotes';

export default {
  title: 'ExperimentView/ExperimentViewNotes',
  component: ExperimentViewNotes,
  argTypes: {},
};

/**
 * Sample redux store
 */
const mockStore = {
  entities: {
    experimentTagsByExperimentId: {
      789: {
        'mlflow.note.content': (ExperimentTag as any).fromJs({
          key: 'mlflow.note.content',
          value: '',
        }),
      },
      1234: {
        'mlflow.note.content': (ExperimentTag as any).fromJs({
          key: 'mlflow.note.content',
          value: 'This is a note!',
        }),
      },
    },
  },
};

/**
 * Sample actions necessary for this component to work
 */
const mockActions: any = {
  setExperimentTagApi: (...args: any) => {
    window.console.log('setExperimentTagApi called with args', args);
    return { type: 'foobar', payload: Promise.resolve('foobar') };
  },
};

const createComponentWrapper = (experiment: Partial<ExperimentEntity>) => () =>
  (
    <Provider store={createStore((s) => s as any, mockStore, compose(applyMiddleware(promiseMiddleware())))}>
      <IntlProvider locale="en">
        <MemoryRouter initialEntries={['/experiments/1234']}>
          <GetExperimentsContextProvider actions={mockActions}>
            <ExperimentViewNotes experiment={experiment as ExperimentEntity} />
          </GetExperimentsContextProvider>
        </MemoryRouter>
      </IntlProvider>
    </Provider>
  );

/**
 * Story for the experiment with no note
 */
export const EmptyNote = createComponentWrapper({ experimentId: '789' });

/**
 * Story for the experiment with note already set
 */
export const PrefilledNote = createComponentWrapper({ experimentId: '1234' });
