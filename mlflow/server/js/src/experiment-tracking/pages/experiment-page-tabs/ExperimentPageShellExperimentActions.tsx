import { useMemo, useState, type ReactNode } from 'react';
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

/**
 * Everything the experiment title header offers, for the breadcrumb header that replaces it: the trace
 * archival badge, the experiment info popover, the page's own action, and the Edit / Delete menu. They
 * follow the title header's rules, so pages that hide its actions or menu hide them here too.
 */
export const ExperimentPageShellExperimentActions = ({
  experiment,
  onNoteUpdated,
  pageAction,
}: {
  experiment: GetExperimentReturnType;
  onNoteUpdated?: () => void;
  pageAction?: ReactNode;
}) => {
  const [editing, setEditing] = useState(false);
  const { headerActionsHidden } = useHeaderVisibility();
  const { pathname } = useLocation();

  // Same conversion as ExperimentPageHeaderWithDescription: the editor and menu expect numeric timestamps.
  const experimentEntity = useMemo(() => {
    if (!experiment) {
      return null;
    }
    return {
      ...experiment,
      creationTime: Number(experiment.creationTime),
      lastUpdateTime: Number(experiment.lastUpdateTime),
    } as ExperimentEntity;
  }, [experiment]);

  if (!experimentEntity) {
    return <>{pageAction}</>;
  }

  const experimentDescription = experimentEntity.tags?.find((tag) => tag.key === NOTE_CONTENT_TAG)?.value;
  const showManagementMenu = !ROUTES_WITHOUT_MANAGEMENT_MENU.some((route) => matchPath(route, pathname));

  return (
    <>
      <ExperimentViewTraceArchivalBadge experiment={experimentEntity} />
      <ExperimentViewInfoPopover experiment={experimentEntity} />
      {!headerActionsHidden && pageAction}
      {!headerActionsHidden && showManagementMenu && (
        <ExperimentViewManagementMenu experiment={experimentEntity} setEditing={setEditing} />
      )}
      <ExperimentViewMetadataEditor
        experiment={experimentEntity}
        editing={editing}
        setEditing={setEditing}
        onNoteUpdated={onNoteUpdated}
        defaultValue={experimentDescription}
      />
    </>
  );
};
