import type { LoggedModelProto } from '../../types';
import { ExperimentLoggedModelSourceBox } from './ExperimentLoggedModelSourceBox';

/**
 * A cell renderer/wrapper component for displaying the model's source in logged models table.
 */
export const ExperimentLoggedModelTableSourceCell = ({ data }: { data: LoggedModelProto }) => {
  return <ExperimentLoggedModelSourceBox loggedModel={data} />;
};
