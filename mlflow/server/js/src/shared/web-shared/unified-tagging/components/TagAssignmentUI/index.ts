import { TagAssignmentRemoveButtonUI } from './TagAssignmentRemoveButtonUI';
import { TagAssignmentRowContainer } from './TagAssignmentRowContainer';
import { TagAssignmentInput } from '../TagAssignmentField/TagAssignmentInput';
import { TagAssignmentLabel } from '../TagAssignmentLabel';
import { TagAssignmentRow } from '../TagAssignmentRow';

/**
 * Contains pure UI components without any built-in RHF state handling.
 * These can be used without useTagAssignmentForm or TagAssignmentRoot.
 */
export const TagAssignmentUI: {
  RowContainer: typeof TagAssignmentRowContainer;
  Row: typeof TagAssignmentRow;
  Input: typeof TagAssignmentInput;
  Label: typeof TagAssignmentLabel;
  RemoveButton: typeof TagAssignmentRemoveButtonUI;
} = {
  RowContainer: TagAssignmentRowContainer,
  Row: TagAssignmentRow,
  Input: TagAssignmentInput,
  Label: TagAssignmentLabel,
  RemoveButton: TagAssignmentRemoveButtonUI,
};
