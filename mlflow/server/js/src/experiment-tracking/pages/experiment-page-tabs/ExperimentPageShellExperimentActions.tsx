import { useMemo, type ReactNode } from 'react';
import { matchPath, useLocation } from '../../../common/utils/RoutingUtils';
import { ROUTES_WITHOUT_MANAGEMENT_MENU } from '../../components/experiment-page/components/header/ExperimentViewHeader';
import { ExperimentViewInfoPopover } from '../../components/experiment-page/components/header/ExperimentViewInfoPopover';
import { ExperimentViewManagementMenu } from '../../components/experiment-page/components/header/ExperimentViewManagementMenu';
import { ExperimentViewTraceArchivalBadge } from '../../components/experiment-page/components/header/ExperimentViewTraceArchivalBadge';
import { ExperimentViewMetadataEditor } from '../../components/experiment-page/components/ExperimentViewMetadataEditor';
import type { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import type { ExperimentEntity } from '../../types';
import { NOTE_CONTENT_TAG } from '../../utils/NoteUtils';
import { useHeaderVisibility } from './ExperimentPageHeaderVisibilityContext';

type GetExperimentReturnType = ReturnType<typeof useGetExperimentQuery>['data'];

// Same conversion as ExperimentPageHeaderWithDescription: the editor and menu expect numeric timestamps.
const useExperimentEntity = (experiment: GetExperimentReturnType) =>
  useMemo(() => {
    if (!experiment) {
      return null;
    }
    return {
      ...experiment,
      creationTime: Number(experiment.creationTime),
      lastUpdateTime: Number(experiment.lastUpdateTime),
    } as ExperimentEntity;
  }, [experiment]);

/**
 * The experiment title header's controls, for the breadcrumb row that replaces it: the trace archival
 * badge, the experiment info popover, the page's own action, and the Edit / Delete menu. They follow the
 * title header's rules, so pages that hide its actions or menu hide them here too.
 */
export const ExperimentPageShellExperimentActions = ({
  experiment,
  setEditing,
  pageAction,
}: {
  experiment: GetExperimentReturnType;
  setEditing: (editing: boolean) => void;
  pageAction?: ReactNode;
}) => {
  const experimentEntity = useExperimentEntity(experiment);
  const { headerActionsHidden } = useHeaderVisibility();
  const { pathname } = useLocation();

  if (!experimentEntity) {
    return <>{pageAction}</>;
  }

  const showManagementMenu = !ROUTES_WITHOUT_MANAGEMENT_MENU.some((route) => matchPath(route, pathname));

  return (
    <>
      <ExperimentViewTraceArchivalBadge experiment={experimentEntity} />
      <ExperimentViewInfoPopover experiment={experimentEntity} />
      {!headerActionsHidden && pageAction}
      {!headerActionsHidden && showManagementMenu && (
        <ExperimentViewManagementMenu experiment={experimentEntity} setEditing={setEditing} />
      )}
    </>
  );
};

/**
 * The experiment description and its editor, shown under the breadcrumb row the way the title header
 * shows them under the title. The editor is shared with the Edit experiment menu item.
 */
export const ExperimentPageShellExperimentDescription = ({
  experiment,
  editing,
  setEditing,
  onNoteUpdated,
}: {
  experiment: GetExperimentReturnType;
  editing: boolean;
  setEditing: (editing: boolean) => void;
  onNoteUpdated?: () => void;
}) => {
  const experimentEntity = useExperimentEntity(experiment);

  if (!experimentEntity) {
    return null;
  }

  const experimentDescription = experimentEntity.tags?.find((tag) => tag.key === NOTE_CONTENT_TAG)?.value;

  return (
    // Without a description only the editor dialog is needed; hiding the wrapper keeps the editor's divider
    // out of the page, and the dialog still opens because it renders in a portal.
    <div css={experimentDescription ? undefined : { display: 'none' }}>
      <ExperimentViewMetadataEditor
        experiment={experimentEntity}
        editing={editing}
        setEditing={setEditing}
        onNoteUpdated={onNoteUpdated}
        defaultValue={experimentDescription}
      />
    </div>
  );
};
