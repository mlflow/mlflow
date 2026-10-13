import type { ReactNode } from 'react';
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

import { validateTemplate } from './agent/validateA2uiMessages';
import { type CustomView, MAX_CUSTOM_VIEWS_PER_EXPERIMENT } from './customViewDefinition';

export type CustomViewDefinitionContextValue = {
  // Every view available for this experiment (persisted views + in-memory
  // edits + a freshly generated, not-yet-saved view).
  views: CustomView[];
  // The currently selected view id (undefined while a brand-new view is being
  // drafted, or before anything is selected).
  activeViewId?: string;
  // The currently selected view, or undefined (empty-state / draft).
  activeView?: CustomView;
  // True while the user has started "Create trace view" but the assistant
  // hasn't produced the first spec yet (the empty-state textbox is showing).
  isDraft: boolean;
  // The name chosen in the naming modal for the in-progress draft.
  draftName: string;
  // Whether the persisted views have finished loading.
  isLoaded: boolean;
  // Whether the experiment already holds the maximum number of saved views. This disables new
  // view creation while keeping edit, rename, and delete available for existing views.
  hasReachedViewLimit: boolean;
  // Whether the user can write custom views to experiment tags (persist callbacks
  // wired AND the host reports experiment edit permission). False for read-only
  // users and the session-local fallback outside the experiment provider.
  canPersist: boolean;
  // Whether persisted views can be deleted (delete callback wired AND the host
  // reports experiment edit permission).
  canDelete: boolean;
  isSaving: boolean;
  // Whether the active view differs from its persisted counterpart (or has none).
  isDirty: boolean;
  // Whether the active view has been persisted to the backend (false for a
  // freshly built, never-saved view).
  isActivePersisted: boolean;
  // Whether the active view's LAST-SAVED (persisted) definition is unreadable —
  // its stored tag couldn't be parsed into a valid view, OR it parsed but its
  // template fails validation. Rename/save rewrite that persisted definition, so
  // both are blocked while this is true to avoid re-persisting the invalid
  // definition; the user must rebuild via the assistant (which clears
  // `unreadable`) and Save first. Keyed on the persisted copy, not the working
  // view, so a rebuilt-but-unsaved view stays blocked until saved.
  isActivePersistedUnreadable: boolean;
  // Ids of every view with unsaved changes: a never-persisted view (new /
  // untitled) or a persisted view whose content has been edited. Drives the
  // per-view "(in draft state)" labeling in the switcher.
  draftViewIds: Set<string>;
  // Whether an initial build (from the empty-state box) is in progress. Lives
  // here (above the host) so it survives the host remount that opening the
  // assistant panel triggers.
  isBuilding: boolean;
  saveError?: string;
  selectView: (id: string) => void;
  // Begin a brand-new view with the given user-provided name. Clears the active
  // selection so the empty-state prompt box shows; the name rides along until
  // the assistant materializes the view.
  startNewView: (name: string) => void;
  // Create or replace a view's content (instruction / template / LLM label). The
  // selection moves to this view only when nothing is selected (a brand-new
  // build) or it is already the active view (an in-place edit); if the user has
  // navigated to a different view while the agent was running, the content is
  // applied in the background and their selection is left untouched. The
  // user-provided `name` is preserved, never overwritten by the assistant.
  // Returns whether the content was written: false means the write was refused
  // because the view was deleted in this session, so callers driving it from an
  // async source can surface the failure instead of assuming it landed.
  upsertViewContent: (view: CustomView) => boolean;
  // Persist the active view, optionally renaming it first (the first-save flow
  // passes the name collected from the modal). Resolves when the persist settles;
  // callers may ignore the returned promise.
  saveActiveView: (nameOverride?: string) => Promise<void>;
  // Rename a persisted view (metadata only): persists the new name against the
  // view's LAST-SAVED template, never the working copy, so an unsaved template
  // edit is left untouched (and the view stays dirty). Distinct from
  // saveActiveView, which commits the full dirty working copy. Resolves when the
  // persist settles; callers may ignore the returned promise.
  renameView: (id: string, name: string) => Promise<void>;
  deleteView: (id: string) => Promise<void>;
  // Mark that an initial build has been launched (shows the building skeleton
  // until the spec applies or the build errors).
  startBuilding: () => void;
  // Clear the building state (spec applied, build failed, or reset).
  stopBuilding: () => void;
};

const CustomViewDefinitionContext = createContext<CustomViewDefinitionContextValue | undefined>(undefined);

// Whether a stored template passes validation. Validation is deferred to here
// (not run at load) so tab load doesn't validate every saved view.
const isTemplateReadable = (template: CustomView['template']): boolean => {
  try {
    return validateTemplate(template).ok;
  } catch {
    return false;
  }
};

// Whether a view can be rendered/edited as-is: it was NOT flagged unreadable at
// load (Case 1 — unparseable/shape-mismatch bytes) and is either an empty draft
// or has a template that validates (Case 2 — derived lazily here). Named
// distinctly from the loader's `view.unreadable` field, which only covers Case 1.
const isViewRenderable = (view: CustomView): boolean =>
  !view.unreadable && (view.template.length === 0 || isTemplateReadable(view.template));

// Upsert a view by id into a list, preserving order (replace in place or append).
const upsertById = (views: CustomView[], view: CustomView): CustomView[] => {
  const index = views.findIndex((entry) => entry.id === view.id);
  if (index < 0) {
    return [...views, view];
  }
  const next = [...views];
  next[index] = view;
  return next;
};

// Core state hook shared by the persistent provider and the session-local
// fallback. It manages the working view registry + active selection, tracks
// per-view dirtiness against the last persisted snapshot, and (when given)
// delegates persistence to `onPersistView` / `onDeleteView`.
export const useCustomViewDefinitionState = (
  initialViews: CustomView[],
  isLoaded: boolean,
  onPersistView?: (view: CustomView) => Promise<void>,
  canModifyPersistedViews?: boolean,
  onDeleteView?: (id: string) => Promise<void>,
  autoSelectFirstView?: boolean,
): CustomViewDefinitionContextValue => {
  const canPersist = Boolean(onPersistView) && Boolean(canModifyPersistedViews);
  const canDelete = Boolean(onDeleteView) && Boolean(canModifyPersistedViews);
  const [views, setViews] = useState<CustomView[]>(initialViews);
  const [persistedViews, setPersistedViews] = useState<CustomView[]>(initialViews);
  // No view is selected by default: the host shows a "Select a custom view"
  // placeholder until the user explicitly picks one. Once picked, the selection
  // lives here (above the drawer) and persists across trace cycling, so the host
  // re-binds the same view's template to each newly opened trace.
  const [activeViewId, setActiveViewId] = useState<string | undefined>(undefined);
  const [isDraft, setIsDraft] = useState(false);
  const [draftName, setDraftName] = useState('');
  // Count of in-flight persistence mutations (save / rename / delete) rather than a
  // boolean: with a shared boolean, whichever of two overlapping mutations
  // finished first would clear the flag while the other is still writing,
  // re-enabling the UI mid-flight. isSaving stays true until the LAST one settles.
  const [savingCount, setSavingCount] = useState(0);
  const isSaving = savingCount > 0;
  const beginSaving = useCallback(() => setSavingCount((count) => count + 1), []);
  const endSaving = useCallback(() => setSavingCount((count) => count - 1), []);
  const [saveError, setSaveError] = useState<string | undefined>(undefined);
  const [isBuilding, setIsBuilding] = useState(false);

  // Ids of views deleted in this session. `upsertViewContent` refuses them, so a
  // write that was already in flight when the user deleted its target cannot
  // bring the view back. A ref is sufficient because nothing renders from it and
  // every reader is a callback that must see the latest set.
  const deletedViewIdsRef = useRef<Set<string>>(new Set());

  // Keep the working set in sync with `initialViews` while the session is still
  // pristine — this adopts views that load ASYNCHRONOUSLY (arriving as a new
  // `initialViews` reference after mount, even if `isLoaded` was already true).
  // Once the user diverges (creates / saves / deletes a view) we stop, so a later refetch
  // can't clobber unsaved local edits. The active selection is left untouched
  // unless `autoSelectFirstView` is set (see the auto-select effect below).
  //
  // We only re-adopt when `initialViews` is a genuinely new reference (callers
  // must pass a stable/memoized array): setting state on every render would loop.
  const hasLocalEditsRef = useRef(false);
  const lastAdoptedRef = useRef<CustomView[] | undefined>(undefined);
  useEffect(() => {
    if (isLoaded && !hasLocalEditsRef.current && initialViews !== lastAdoptedRef.current) {
      lastAdoptedRef.current = initialViews;
      setViews(initialViews);
      setPersistedViews(initialViews);
    }
  }, [isLoaded, initialViews]);

  const activeView = useMemo(() => views.find((view) => view.id === activeViewId), [views, activeViewId]);

  const activeViewIdRef = useRef(activeViewId);
  useEffect(() => {
    activeViewIdRef.current = activeViewId;
  }, [activeViewId]);

  // When `autoSelectFirstView` is set, auto-select the first persisted view on
  // first load so a saved view renders immediately instead of the "select a
  // view" placeholder the host shows when views exist but none is chosen. With
  // no saved views this is a no-op (the host renders its create-first-view
  // authoring empty state instead). Skip once the user has diverged (draft,
  // delete, explicit selection) so we never steal a later empty state.
  useEffect(() => {
    if (
      !autoSelectFirstView ||
      !isLoaded ||
      hasLocalEditsRef.current ||
      isDraft ||
      activeViewIdRef.current !== undefined
    ) {
      return;
    }
    const firstId = views[0]?.id;
    if (firstId) {
      setActiveViewId(firstId);
    }
  }, [autoSelectFirstView, isLoaded, views, isDraft]);

  const isDirty = useMemo(() => {
    if (!activeView) {
      return false;
    }
    const persisted = persistedViews.find((view) => view.id === activeView.id);
    return JSON.stringify(activeView) !== JSON.stringify(persisted);
  }, [activeView, persistedViews]);

  const isActivePersisted = useMemo(
    () => Boolean(activeViewId) && persistedViews.some((view) => view.id === activeViewId),
    [activeViewId, persistedViews],
  );

  const isActivePersistedUnreadable = useMemo(() => {
    const persisted = persistedViews.find((view) => view.id === activeViewId);
    return persisted ? !isViewRenderable(persisted) : false;
  }, [activeViewId, persistedViews]);

  const hasReachedViewLimit = persistedViews.length >= MAX_CUSTOM_VIEWS_PER_EXPERIMENT;

  const draftViewIds = useMemo(() => {
    const ids = new Set<string>();
    for (const view of views) {
      const persisted = persistedViews.find((v) => v.id === view.id);
      if (!persisted || JSON.stringify(view) !== JSON.stringify(persisted)) {
        ids.add(view.id);
      }
    }
    return ids;
  }, [views, persistedViews]);

  const startBuilding = useCallback(() => setIsBuilding(true), []);
  const stopBuilding = useCallback(() => setIsBuilding(false), []);

  const selectView = useCallback((id: string) => {
    setSaveError(undefined);
    setIsDraft(false);
    setDraftName('');
    setActiveViewId(id);
  }, []);

  const startNewView = useCallback(
    (name: string) => {
      if (!canPersist || hasReachedViewLimit) {
        return;
      }
      setSaveError(undefined);
      setDraftName(name);
      setIsDraft(true);
      setActiveViewId(undefined);
    },
    [canPersist, hasReachedViewLimit],
  );

  const upsertViewContent = useCallback(
    (view: CustomView) => {
      if (!canPersist) {
        return false;
      }
      if (deletedViewIdsRef.current.has(view.id)) {
        return false;
      }
      hasLocalEditsRef.current = true;
      // Always write the content: this is the actual apply, and is all that's
      // needed for the view to exist / update in the working set (the switcher and
      // the per-trace re-bind read from `views`). The name is USER-owned and the
      // agent never authors it, so for an existing view we keep whatever name is
      // current — a rename that landed while the agent request was in flight must
      // not be reverted by the stale name captured when the edit was launched. A
      // brand-new view (no existing entry) uses the incoming (draft) name.
      setViews((prev) => {
        const existing = prev.find((entry) => entry.id === view.id);
        return upsertById(prev, existing ? { ...view, name: existing.name } : view);
      });
      // Only move the selection when it makes sense: nothing is selected yet (a
      // brand-new build — showing the result is the point) or this is already the
      // active view (a normal in-place edit). If the user navigated to a different
      // view while the agent was running, leave their selection and any in-progress
      // draft untouched so the result applies silently in the background.
      const shouldSelect = activeViewIdRef.current === undefined || activeViewIdRef.current === view.id;
      if (shouldSelect) {
        setActiveViewId(view.id);
        setIsDraft(false);
        setDraftName('');
      }
      return true;
    },
    [canPersist],
  );

  const saveActiveView = useCallback(
    async (nameOverride?: string) => {
      if (!canPersist || !onPersistView || !activeView) {
        return;
      }
      // An unreadable view must not be round-tripped through persist: a Case-1
      // placeholder would overwrite the original stored bytes with the empty
      // synthesized definition, and a Case-2 invalid template would re-persist a
      // definition that can't render. The user must rebuild via the assistant first.
      if (!isViewRenderable(activeView)) {
        return;
      }
      const view =
        nameOverride !== undefined && nameOverride.trim() ? { ...activeView, name: nameOverride.trim() } : activeView;
      beginSaving();
      setSaveError(undefined);
      try {
        await onPersistView(view);
        hasLocalEditsRef.current = true;
        // Propagate only the (possibly renamed) NAME into the live working copy,
        // never the pre-await content snapshot: an edit that landed during the
        // await (e.g. an assistant apply via upsertViewContent) must not be
        // reverted. persistedViews records what we actually wrote, so a concurrent
        // edit stays correctly dirty instead of being silently clobbered and marked
        // clean.
        setViews((prev) => prev.map((entry) => (entry.id === view.id ? { ...entry, name: view.name } : entry)));
        setPersistedViews((prev) => upsertById(prev, view));
      } catch (error) {
        setSaveError(error instanceof Error ? error.message : 'Failed to save the custom view.');
      } finally {
        endSaving();
      }
    },
    [canPersist, onPersistView, activeView, beginSaving, endSaving],
  );

  const renameView = useCallback(
    async (id: string, name: string) => {
      const trimmed = name.trim();
      if (!canPersist || !onPersistView || !trimmed) {
        return;
      }
      // Rename against the LAST-SAVED snapshot, never the working copy, so an
      // unsaved template edit is not silently committed by a metadata rename.
      const base = persistedViews.find((view) => view.id === id);
      if (!base || trimmed === base.name) {
        return;
      }
      // Rename rewrites the persisted definition. If that definition is unreadable
      // (Case 1: stored bytes couldn't be parsed; Case 2: its template fails
      // validation), rewriting it would destroy the original bytes / re-persist an
      // unrenderable definition — so block the rename until the view is rebuilt via
      // the assistant (which produces a valid template) and saved.
      if (!isViewRenderable(base)) {
        return;
      }
      const renamed = { ...base, name: trimmed };
      beginSaving();
      setSaveError(undefined);
      try {
        await onPersistView(renamed);
        hasLocalEditsRef.current = true;
        // Propagate ONLY the name into both the working copy and the persisted
        // snapshot, via functional updaters over the LATEST state — never write
        // back the pre-await `renamed` object. A concurrent edit
        // (upsertViewContent) keeps its working template and stays dirty, and a
        // concurrent save that landed a newer persisted template during the
        // await is not rolled back to the old one; the rename only relabels
        // whatever is current.
        setViews((prev) => prev.map((entry) => (entry.id === id ? { ...entry, name: trimmed } : entry)));
        setPersistedViews((prev) => prev.map((entry) => (entry.id === id ? { ...entry, name: trimmed } : entry)));
      } catch (error) {
        setSaveError(error instanceof Error ? error.message : 'Failed to rename the custom view.');
      } finally {
        endSaving();
      }
    },
    [canPersist, onPersistView, persistedViews, beginSaving, endSaving],
  );

  const deleteView = useCallback(
    async (id: string) => {
      setSaveError(undefined);
      if (!canDelete || !onDeleteView) {
        return;
      }
      beginSaving();
      // Tombstone before the backend request so an Assistant response that lands
      // while deletion is in flight cannot update a view the user chose to delete.
      deletedViewIdsRef.current.add(id);
      try {
        if (persistedViews.some((view) => view.id === id)) {
          await onDeleteView(id);
        }
        hasLocalEditsRef.current = true;
        setPersistedViews((prev) => prev.filter((view) => view.id !== id));
        setViews((prev) => prev.filter((view) => view.id !== id));
        setActiveViewId((current) => (current === id ? undefined : current));
      } catch (error) {
        // The view remains when persistence fails, so allow subsequent updates.
        deletedViewIdsRef.current.delete(id);
        setSaveError(error instanceof Error ? error.message : 'Failed to delete the custom view.');
      } finally {
        endSaving();
      }
    },
    [canDelete, onDeleteView, persistedViews, beginSaving, endSaving],
  );

  return {
    views,
    activeViewId,
    activeView,
    isDraft,
    draftName,
    isLoaded,
    hasReachedViewLimit,
    canPersist,
    canDelete,
    isSaving,
    isDirty,
    isActivePersisted,
    isActivePersistedUnreadable,
    draftViewIds,
    isBuilding,
    saveError,
    selectView,
    startNewView,
    upsertViewContent,
    saveActiveView,
    renameView,
    deleteView,
    startBuilding,
    stopBuilding,
  };
};

// Generic provider: experiment-tracking wires this up with the loaded views +
// persist/delete callbacks that write per-view experiment tags. Mounted high enough
// (e.g. in the traces table) that it survives drawer close / trace cycling.
export const CustomViewDefinitionProvider = ({
  views,
  isLoaded,
  onPersistView,
  onDeleteView,
  canModifyPersistedViews,
  autoSelectFirstView,
  children,
}: {
  views: CustomView[];
  isLoaded: boolean;
  onPersistView?: (view: CustomView) => Promise<void>;
  onDeleteView?: (id: string) => Promise<void>;
  canModifyPersistedViews?: boolean;
  autoSelectFirstView?: boolean;
  children: ReactNode;
}): JSX.Element => {
  const value = useCustomViewDefinitionState(
    views,
    isLoaded,
    onPersistView,
    canModifyPersistedViews,
    onDeleteView,
    autoSelectFirstView,
  );
  return <CustomViewDefinitionContext.Provider value={value}>{children}</CustomViewDefinitionContext.Provider>;
};

// Returns the experiment-scoped value, or undefined when no provider is mounted
// (e.g. a standalone embed). Callers fall back to a session-local state hook.
export const useOptionalCustomViewDefinition = (): CustomViewDefinitionContextValue | undefined =>
  useContext(CustomViewDefinitionContext);

const EMPTY_VIEWS: CustomView[] = [];

// Resolves the active custom-view state, preferring the experiment-scoped
// provider when present and otherwise falling back to a session-local (non
// -persisting) state engine. Both hooks always run (the session-local one is a
// no-op when the provider is mounted), keeping the hook order stable.
export const useCustomViewDefinition = (): CustomViewDefinitionContextValue => {
  const provided = useOptionalCustomViewDefinition();
  const sessionLocal = useCustomViewDefinitionState(EMPTY_VIEWS, true);
  return provided ?? sessionLocal;
};
