/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { ExperimentPagePersistedState } from './MlflowLocalStorageMessages';

test('Local storage messages ignore unknown fields', () => {
  const persistedState = ExperimentPagePersistedState({ heyYallImAnUnknownField: 'value' });
  expect((persistedState as any).searchInput).toEqual('');
});
