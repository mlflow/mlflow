import { describe, it, expect, jest } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';

import * as validateModule from './agent/validateA2uiMessages';
import { useCustomViewDefinitionState } from './CustomViewDefinitionContext';
import { type CustomView, MAX_CUSTOM_VIEWS_PER_EXPERIMENT } from './customViewDefinition';

const makeView = (id: string, overrides: Partial<CustomView> = {}): CustomView => ({
  id,
  name: `name-${id}`,
  label: `label-${id}`,
  instruction: `do ${id}`,
  template: [],
  createdAtMs: 1,
  ...overrides,
});

// The hook adopts `initialViews` only when it's a genuinely new reference, so
// tests must pass a stable array (an inline `[]` would be a new ref every render
// and re-trigger adoption). These module-level fixtures give stable references.
const NO_VIEWS: CustomView[] = [];
// A single persisted view as a stable reference — rename tests start from a
// populated `views` + `persistedViews` (both seeded from `initialViews`).
const SINGLE_VIEW: CustomView[] = [makeView('a')];
const noopPersistView = (_view: CustomView): Promise<void> => Promise.resolve();

describe('useCustomViewDefinitionState', () => {
  it('starts with no active selection and reports canPersist from the persist callback', () => {
    const { result } = renderHook(() => useCustomViewDefinitionState(NO_VIEWS, true));

    expect(result.current.views).toEqual([]);
    expect(result.current.activeViewId).toBeUndefined();
    expect(result.current.activeView).toBeUndefined();
    expect(result.current.isDraft).toBe(false);
    expect(result.current.canPersist).toBe(false);

    const withPersist = renderHook(() =>
      useCustomViewDefinitionState(NO_VIEWS, true, jest.fn<() => Promise<void>>(), true),
    );
    expect(withPersist.result.current.canPersist).toBe(true);
    expect(withPersist.result.current.canDelete).toBe(false);

    const withDelete = renderHook(() =>
      useCustomViewDefinitionState(NO_VIEWS, true, noopPersistView, true, jest.fn<() => Promise<void>>()),
    );
    expect(withDelete.result.current.canDelete).toBe(true);
  });

  it('reports canPersist false when a persist callback exists but edit permission is denied', () => {
    const { result } = renderHook(() =>
      useCustomViewDefinitionState(
        NO_VIEWS,
        true,
        jest.fn<() => Promise<void>>(),
        false,
        jest.fn<() => Promise<void>>(),
      ),
    );
    expect(result.current.canPersist).toBe(false);
    expect(result.current.canDelete).toBe(false);
  });

  it('does not create or update working views when modification permission is denied', () => {
    const initialViews = [makeView('a')];
    const { result } = renderHook(() =>
      useCustomViewDefinitionState(initialViews, true, jest.fn<() => Promise<void>>(), false),
    );

    act(() => result.current.startNewView('Blocked draft'));
    act(() => result.current.upsertViewContent(makeView('b')));

    expect(result.current.isDraft).toBe(false);
    expect(result.current.views).toEqual(initialViews);
  });

  it('does not create working views when no persistence callback is available', () => {
    const { result } = renderHook(() => useCustomViewDefinitionState(NO_VIEWS, true));

    act(() => result.current.startNewView('Blocked draft'));
    act(() => result.current.upsertViewContent(makeView('a')));

    expect(result.current.isDraft).toBe(false);
    expect(result.current.views).toEqual([]);
  });

  it('upserts a new view, selects it, and reports it dirty until persisted', () => {
    const { result } = renderHook(() => useCustomViewDefinitionState(NO_VIEWS, true, noopPersistView, true));

    act(() => result.current.upsertViewContent(makeView('a')));

    expect(result.current.activeViewId).toBe('a');
    expect(result.current.activeView).toEqual(makeView('a'));
    expect(result.current.isDraft).toBe(false);
    // Never-persisted view: dirty against its (absent) persisted snapshot.
    expect(result.current.isDirty).toBe(true);
    expect(result.current.isActivePersisted).toBe(false);
  });

  it('transitions between startNewView (draft), upsert, and selectView', () => {
    const { result } = renderHook(() => useCustomViewDefinitionState(NO_VIEWS, true, noopPersistView, true));

    act(() => result.current.upsertViewContent(makeView('a')));

    // startNewView clears the active selection and opens the draft empty state.
    act(() => result.current.startNewView('My new view'));
    expect(result.current.isDraft).toBe(true);
    expect(result.current.draftName).toBe('My new view');
    expect(result.current.activeViewId).toBeUndefined();

    // Materializing the draft ends the draft state and selects the new view.
    act(() => result.current.upsertViewContent(makeView('b')));
    expect(result.current.isDraft).toBe(false);
    expect(result.current.draftName).toBe('');
    expect(result.current.activeViewId).toBe('b');
    expect(result.current.views.map((view) => view.id)).toEqual(['a', 'b']);

    // Reselecting an existing view exits any draft state.
    act(() => result.current.selectView('a'));
    expect(result.current.activeViewId).toBe('a');
    expect(result.current.isDraft).toBe(false);
  });

  it('applies upserted content in the background without stealing selection when another view is active', () => {
    const initialViews = [makeView('a'), makeView('b')];
    const { result } = renderHook(() => useCustomViewDefinitionState(initialViews, true, noopPersistView, true));

    // User is viewing B when an edit launched for A finally lands.
    act(() => result.current.selectView('b'));
    act(() => result.current.upsertViewContent(makeView('a', { label: 'rebuilt-a' })));

    // A's content is updated in the working set, but the user stays on B.
    expect(result.current.activeViewId).toBe('b');
    expect(result.current.views.find((view) => view.id === 'a')?.label).toBe('rebuilt-a');
    // B is untouched.
    expect(result.current.views.find((view) => view.id === 'b')).toEqual(makeView('b'));
  });

  it('keeps the selection on an in-place edit of the active view', () => {
    const initialViews = [makeView('a')];
    const { result } = renderHook(() => useCustomViewDefinitionState(initialViews, true, noopPersistView, true));

    act(() => result.current.selectView('a'));
    act(() => result.current.upsertViewContent(makeView('a', { label: 'rebuilt-a' })));

    expect(result.current.activeViewId).toBe('a');
    expect(result.current.activeView?.label).toBe('rebuilt-a');
  });

  it('preserves a concurrent edit that lands during an in-flight save instead of reverting it', async () => {
    // Hold the persist promise open so we can inject an edit mid-save.
    let resolvePersist: () => void = () => {};
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>(
      () =>
        new Promise<void>((resolve) => {
          resolvePersist = resolve;
        }),
    );
    const { result } = renderHook(() => useCustomViewDefinitionState(NO_VIEWS, true, onPersistView, true));

    act(() => result.current.upsertViewContent(makeView('a', { label: 'v1' })));

    // `saveActiveView` is typed as returning void (async at runtime), so capture
    // it loosely and await the runtime promise to flush the resolve handlers.
    let savePromise: Promise<void> | void = undefined;
    act(() => {
      savePromise = result.current.saveActiveView();
    });
    expect(result.current.isSaving).toBe(true);

    // A newer edit (e.g. an assistant apply) lands while the save is still in flight.
    act(() => result.current.upsertViewContent(makeView('a', { label: 'v2' })));

    await act(async () => {
      resolvePersist();
      await savePromise;
    });

    // The server received the pre-await snapshot, but the newer edit survives in
    // the working copy and the view is still reported dirty (v2 is unsaved).
    expect(onPersistView).toHaveBeenCalledWith(makeView('a', { label: 'v1' }));
    expect(result.current.activeView?.label).toBe('v2');
    expect(result.current.isDirty).toBe(true);
  });

  it('saveActiveView persists the active view, clearing dirtiness', async () => {
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
    const { result } = renderHook(() => useCustomViewDefinitionState(NO_VIEWS, true, onPersistView, true));

    act(() => result.current.upsertViewContent(makeView('a')));
    expect(result.current.isDirty).toBe(true);
    expect(result.current.isActivePersisted).toBe(false);

    await act(async () => {
      await result.current.saveActiveView();
    });

    expect(onPersistView).toHaveBeenCalledWith(makeView('a'));
    expect(result.current.isActivePersisted).toBe(true);
    expect(result.current.isDirty).toBe(false);
    expect(result.current.isSaving).toBe(false);
    expect(result.current.saveError).toBeUndefined();
  });

  it('saveActiveView applies a trimmed name override before persisting', async () => {
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
    const { result } = renderHook(() => useCustomViewDefinitionState(NO_VIEWS, true, onPersistView, true));

    act(() => result.current.upsertViewContent(makeView('a')));
    await act(async () => {
      await result.current.saveActiveView('  Renamed  ');
    });

    expect(onPersistView).toHaveBeenCalledWith({ ...makeView('a'), name: 'Renamed' });
    expect(result.current.activeView?.name).toBe('Renamed');
  });

  it('saveActiveView surfaces a persist failure and leaves the view dirty', async () => {
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockRejectedValue(new Error('save exploded'));
    const { result } = renderHook(() => useCustomViewDefinitionState(NO_VIEWS, true, onPersistView, true));

    act(() => result.current.upsertViewContent(makeView('a')));
    await act(async () => {
      await result.current.saveActiveView();
    });

    expect(result.current.saveError).toBe('save exploded');
    expect(result.current.isSaving).toBe(false);
    expect(result.current.isActivePersisted).toBe(false);
    expect(result.current.isDirty).toBe(true);
  });

  it('renameView renames a clean persisted view and it stays not dirty', async () => {
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
    const { result } = renderHook(() => useCustomViewDefinitionState(SINGLE_VIEW, true, onPersistView, true));

    act(() => result.current.selectView('a'));
    await act(async () => {
      await result.current.renameView('a', 'New name');
    });

    expect(onPersistView).toHaveBeenCalledWith({ ...makeView('a'), name: 'New name' });
    expect(result.current.activeView?.name).toBe('New name');
    expect(result.current.isActivePersisted).toBe(true);
    expect(result.current.isDirty).toBe(false);
    expect(result.current.saveError).toBeUndefined();
  });

  it('renameView persists the last-saved template under the new (trimmed) name and keeps unsaved edits dirty', async () => {
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
    const editedTemplate: CustomView['template'] = [
      {
        version: 'v0.9',
        updateComponents: { surfaceId: 'main', components: [{ id: 'root', component: 'Text', text: 'edited' }] },
      },
    ];
    const { result } = renderHook(() => useCustomViewDefinitionState(SINGLE_VIEW, true, onPersistView, true));

    act(() => result.current.selectView('a'));
    // Dirty the working copy with a template edit (e.g. an assistant apply).
    act(() => result.current.upsertViewContent(makeView('a', { template: editedTemplate })));
    expect(result.current.isDirty).toBe(true);

    await act(async () => {
      await result.current.renameView('a', '  New name  ');
    });

    // The server received the LAST-SAVED template (makeView('a').template) with
    // only the (trimmed) name changed — not the dirty working template.
    expect(onPersistView).toHaveBeenCalledWith({ ...makeView('a'), name: 'New name' });
    // The working copy keeps the unsaved template edit + the new name, and stays dirty.
    expect(result.current.activeView?.name).toBe('New name');
    expect(result.current.activeView?.template).toBe(editedTemplate);
    expect(result.current.isDirty).toBe(true);
  });

  it('renameView preserves a concurrent edit that lands during the in-flight persist', async () => {
    let resolvePersist: () => void = () => {};
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>(
      () =>
        new Promise<void>((resolve) => {
          resolvePersist = resolve;
        }),
    );
    const { result } = renderHook(() => useCustomViewDefinitionState(SINGLE_VIEW, true, onPersistView, true));

    act(() => result.current.selectView('a'));

    let renamePromise: Promise<void> | void = undefined;
    act(() => {
      renamePromise = result.current.renameView('a', 'Renamed');
    });
    expect(result.current.isSaving).toBe(true);

    // A newer edit (assistant apply) lands while the rename persist is still open.
    act(() => result.current.upsertViewContent(makeView('a', { label: 'v2' })));

    await act(async () => {
      resolvePersist();
      await renamePromise;
    });

    // The server got the pre-edit template + new name; the concurrent edit
    // survives in the working copy and the view is still dirty.
    expect(onPersistView).toHaveBeenCalledWith({ ...makeView('a'), name: 'Renamed' });
    expect(result.current.activeView?.name).toBe('Renamed');
    expect(result.current.activeView?.label).toBe('v2');
    expect(result.current.isDirty).toBe(true);
  });

  it('renameView does not roll back a newer template saved concurrently during its persist', async () => {
    // One controllable promise per persist call, resolved in the order we choose.
    const resolvers: Array<() => void> = [];
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>(
      () =>
        new Promise<void>((resolve) => {
          resolvers.push(resolve);
        }),
    );
    const newTemplate: CustomView['template'] = [
      {
        version: 'v0.9',
        updateComponents: { surfaceId: 'main', components: [{ id: 'root', component: 'Text', text: 'newer' }] },
      },
    ];
    const { result } = renderHook(() => useCustomViewDefinitionState(SINGLE_VIEW, true, onPersistView, true));

    act(() => result.current.selectView('a'));
    // Dirty the working copy with a newer template, then start saving it.
    act(() => result.current.upsertViewContent(makeView('a', { template: newTemplate })));

    let savePromise: Promise<void> | void = undefined;
    act(() => {
      savePromise = result.current.saveActiveView();
    });
    // The rename launches while the save is still in flight, so it reads the
    // OLD persisted snapshot (the pre-save template) as its base.
    let renamePromise: Promise<void> | void = undefined;
    act(() => {
      renamePromise = result.current.renameView('a', 'Renamed');
    });

    // Resolve the save first — it persists the newer template — then the rename.
    await act(async () => {
      resolvers[0]();
      await savePromise;
    });
    await act(async () => {
      resolvers[1]();
      await renamePromise;
    });

    // The concurrently-saved newer template must survive: the rename only
    // relabels the LATEST persisted snapshot, so the view is renamed AND clean
    // (not rolled back to the pre-save template, which would read as dirty).
    expect(result.current.activeView?.name).toBe('Renamed');
    expect(result.current.activeView?.template).toBe(newTemplate);
    expect(result.current.isDirty).toBe(false);
  });

  it('upsertViewContent never overwrites an existing view name, so a rename is not reverted by an in-flight agent edit', async () => {
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
    const { result } = renderHook(() => useCustomViewDefinitionState(SINGLE_VIEW, true, onPersistView, true));

    act(() => result.current.selectView('a'));

    // The user renames the view (e.g. while an assistant draft edit is in flight).
    await act(async () => {
      await result.current.renameView('a', 'Renamed');
    });
    expect(result.current.activeView?.name).toBe('Renamed');

    // The in-flight agent edit finally lands, carrying the STALE launch-time name
    // (name-a) alongside its new template. The name must not be reverted.
    const newTemplate: CustomView['template'] = [
      {
        version: 'v0.9',
        updateComponents: { surfaceId: 'main', components: [{ id: 'root', component: 'Text', text: 'agent' }] },
      },
    ];
    act(() => result.current.upsertViewContent(makeView('a', { name: 'name-a', template: newTemplate })));

    // The user's rename survives; the agent's template still applies.
    expect(result.current.activeView?.name).toBe('Renamed');
    expect(result.current.activeView?.template).toBe(newTemplate);
  });

  it('renameView surfaces a persist failure and leaves the name unchanged', async () => {
    const onPersistView = jest
      .fn<(view: CustomView) => Promise<void>>()
      .mockRejectedValue(new Error('rename exploded'));
    const { result } = renderHook(() => useCustomViewDefinitionState(SINGLE_VIEW, true, onPersistView, true));

    act(() => result.current.selectView('a'));
    await act(async () => {
      await result.current.renameView('a', 'New name');
    });

    expect(result.current.saveError).toBe('rename exploded');
    expect(result.current.activeView?.name).toBe('name-a');
    expect(result.current.isSaving).toBe(false);
  });

  it('renameView is a no-op without modification permission or a persist callback', async () => {
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
    const denied = renderHook(() => useCustomViewDefinitionState(SINGLE_VIEW, true, onPersistView, false));

    act(() => denied.result.current.selectView('a'));
    await act(async () => {
      await denied.result.current.renameView('a', 'New name');
    });
    expect(onPersistView).not.toHaveBeenCalled();
    expect(denied.result.current.activeView?.name).toBe('name-a');

    const noCallback = renderHook(() => useCustomViewDefinitionState(SINGLE_VIEW, true));
    act(() => noCallback.result.current.selectView('a'));
    await act(async () => {
      await noCallback.result.current.renameView('a', 'New name');
    });
    expect(noCallback.result.current.activeView?.name).toBe('name-a');
  });

  it('renameView ignores an empty/whitespace name and a no-op same-name rename', async () => {
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
    const { result } = renderHook(() => useCustomViewDefinitionState(SINGLE_VIEW, true, onPersistView, true));

    act(() => result.current.selectView('a'));

    await act(async () => {
      await result.current.renameView('a', '   ');
    });
    // Renaming to the current name is a no-op write too.
    await act(async () => {
      await result.current.renameView('a', 'name-a');
    });

    expect(onPersistView).not.toHaveBeenCalled();
    expect(result.current.activeView?.name).toBe('name-a');
  });

  it('blocks renameView for an unreadable persisted view so the original stored bytes are not clobbered', async () => {
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
    const initialViews = [makeView('a', { unreadable: true })];
    const { result } = renderHook(() => useCustomViewDefinitionState(initialViews, true, onPersistView, true));

    act(() => result.current.selectView('a'));
    expect(result.current.isActivePersistedUnreadable).toBe(true);

    await act(async () => {
      await result.current.renameView('a', 'New name');
    });

    expect(onPersistView).not.toHaveBeenCalled();
    expect(result.current.activeView?.name).toBe('name-a');
  });

  it('blocks saveActiveView for an unreadable active view', async () => {
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
    const initialViews = [makeView('a', { unreadable: true })];
    const { result } = renderHook(() => useCustomViewDefinitionState(initialViews, true, onPersistView, true));

    act(() => result.current.selectView('a'));
    await act(async () => {
      await result.current.saveActiveView('New name');
    });

    expect(onPersistView).not.toHaveBeenCalled();
  });

  it('re-enables rename after an unreadable view is rebuilt and saved', async () => {
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
    const initialViews = [makeView('a', { unreadable: true })];
    const validTemplate: CustomView['template'] = [
      {
        version: 'v0.9',
        updateComponents: { surfaceId: 'main', components: [{ id: 'root', component: 'Text', text: 'rebuilt' }] },
      },
    ];
    const { result } = renderHook(() => useCustomViewDefinitionState(initialViews, true, onPersistView, true));

    act(() => result.current.selectView('a'));
    expect(result.current.isActivePersistedUnreadable).toBe(true);

    // Rebuild via the agent apply path: the new content carries no `unreadable`
    // flag, so the working view is readable — but the persisted copy is still the
    // placeholder, so rename stays blocked until the rebuild is saved.
    act(() => result.current.upsertViewContent(makeView('a', { template: validTemplate })));
    expect(result.current.isActivePersistedUnreadable).toBe(true);

    await act(async () => {
      await result.current.saveActiveView();
    });
    // The persisted copy is now the valid rebuilt view → rename is re-enabled.
    expect(result.current.isActivePersistedUnreadable).toBe(false);

    await act(async () => {
      await result.current.renameView('a', 'Renamed');
    });
    expect(onPersistView).toHaveBeenCalledWith(expect.objectContaining({ id: 'a', name: 'Renamed' }));
    expect(result.current.activeView?.name).toBe('Renamed');
  });

  it('treats a template whose validation throws as unreadable instead of crashing', () => {
    // isViewRenderable runs validateTemplate on render/menu/save/rename paths that
    // have no surrounding containment. validateTemplate walks untrusted stored
    // template data, so an unforeseen throw must degrade to a Case-2 unreadable
    // placeholder (blocked rename/save) rather than propagate and crash the UI.
    // The template below is otherwise VALID (has a root), so `unreadable` can only
    // be true via the throw-containment path — not because the template is invalid.
    const validTemplate: CustomView['template'] = [
      {
        version: 'v0.9',
        updateComponents: { surfaceId: 'main', components: [{ id: 'root', component: 'Text', text: 'hi' }] },
      },
    ];
    const spy = jest.spyOn(validateModule, 'validateTemplate').mockImplementation(() => {
      throw new Error('hostile template');
    });
    try {
      const initialViews = [makeView('a', { template: validTemplate })];
      const { result } = renderHook(() => useCustomViewDefinitionState(initialViews, true, noopPersistView, true));

      act(() => result.current.selectView('a'));
      expect(result.current.isActivePersistedUnreadable).toBe(true);
      expect(spy).toHaveBeenCalled();
    } finally {
      spy.mockRestore();
    }
  });

  it('deletes a persisted view and clears the active selection', async () => {
    const onDeleteView = jest.fn<(id: string) => Promise<void>>().mockResolvedValue(undefined);
    const initialViews = [makeView('a'), makeView('b')];
    const { result } = renderHook(() =>
      useCustomViewDefinitionState(initialViews, true, noopPersistView, true, onDeleteView),
    );

    act(() => result.current.selectView('a'));
    await act(async () => {
      await result.current.deleteView('a');
    });

    expect(onDeleteView).toHaveBeenCalledWith('a');
    expect(result.current.views.map((view) => view.id)).toEqual(['b']);
    expect(result.current.activeViewId).toBeUndefined();
  });

  it('keeps the view and surfaces the error when deletion fails', async () => {
    const onDeleteView = jest.fn<(id: string) => Promise<void>>().mockRejectedValue(new Error('delete exploded'));
    const { result } = renderHook(() =>
      useCustomViewDefinitionState(SINGLE_VIEW, true, noopPersistView, true, onDeleteView),
    );

    act(() => result.current.selectView('a'));
    await act(async () => {
      await result.current.deleteView('a');
    });

    expect(result.current.saveError).toBe('delete exploded');
    expect(result.current.views).toEqual(SINGLE_VIEW);
    expect(result.current.activeViewId).toBe('a');
    expect(result.current.isSaving).toBe(false);

    let applied: boolean | undefined;
    act(() => {
      applied = result.current.upsertViewContent(makeView('a', { label: 'updated' }));
    });
    expect(applied).toBe(true);
    expect(result.current.activeView?.label).toBe('updated');
  });

  it('rejects an update that lands while deletion is in flight', async () => {
    let resolveDelete: () => void = () => {};
    const onDeleteView = jest.fn<(id: string) => Promise<void>>(
      () =>
        new Promise<void>((resolve) => {
          resolveDelete = resolve;
        }),
    );
    const { result } = renderHook(() =>
      useCustomViewDefinitionState(SINGLE_VIEW, true, noopPersistView, true, onDeleteView),
    );

    act(() => result.current.selectView('a'));
    let deletePromise: Promise<void> | undefined;
    act(() => {
      deletePromise = result.current.deleteView('a');
    });

    let applied: boolean | undefined;
    act(() => {
      applied = result.current.upsertViewContent(makeView('a'));
    });

    expect(applied).toBe(false);
    expect(result.current.views).toEqual(SINGLE_VIEW);

    await act(async () => {
      resolveDelete();
      await deletePromise;
    });

    expect(result.current.views).toEqual([]);
    expect(result.current.activeViewId).toBeUndefined();
  });

  it('keeps isSaving true until the LAST of two overlapping mutations settles', async () => {
    // Two independently controllable persists so we can settle them out of order.
    const resolvers: Array<() => void> = [];
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>(
      () =>
        new Promise<void>((resolve) => {
          resolvers.push(resolve);
        }),
    );
    const { result } = renderHook(() => useCustomViewDefinitionState(SINGLE_VIEW, true, onPersistView, true));

    act(() => result.current.selectView('a'));

    // Start a save and a rename that overlap (both in flight at once).
    let savePromise: Promise<void> | void = undefined;
    let renamePromise: Promise<void> | void = undefined;
    act(() => {
      savePromise = result.current.saveActiveView();
    });
    act(() => {
      renamePromise = result.current.renameView('a', 'Renamed');
    });
    expect(result.current.isSaving).toBe(true);

    // First mutation settles — isSaving must STAY true (the other is still writing).
    await act(async () => {
      resolvers[0]();
      await savePromise;
    });
    expect(result.current.isSaving).toBe(true);

    // Last mutation settles — only now does isSaving clear.
    await act(async () => {
      resolvers[1]();
      await renamePromise;
    });
    expect(result.current.isSaving).toBe(false);
  });

  it('adopts asynchronously loaded views once, and a selected loaded view is not dirty', () => {
    const loaded = [makeView('a')];
    const { result, rerender } = renderHook(
      ({ views, isLoaded }: { views: CustomView[]; isLoaded: boolean }) =>
        useCustomViewDefinitionState(views, isLoaded, noopPersistView, true),
      { initialProps: { views: NO_VIEWS, isLoaded: false } },
    );

    expect(result.current.views).toEqual([]);

    // Views arrive after mount (new reference, session still pristine).
    rerender({ views: loaded, isLoaded: true });
    expect(result.current.views).toEqual(loaded);

    // The adopted view is its own persisted snapshot, so it is clean.
    act(() => result.current.selectView('a'));
    expect(result.current.isDirty).toBe(false);
    expect(result.current.isActivePersisted).toBe(true);
  });

  describe('per-experiment view limit', () => {
    const MAX_VIEWS: CustomView[] = Array.from({ length: MAX_CUSTOM_VIEWS_PER_EXPERIMENT }, (_unused, index) =>
      makeView(`view-${index}`),
    );

    it('reports the limit only after persisted views reach the cap', () => {
      const belowLimit = MAX_VIEWS.slice(0, MAX_CUSTOM_VIEWS_PER_EXPERIMENT - 1);
      const below = renderHook(() => useCustomViewDefinitionState(belowLimit, true, noopPersistView, true));
      expect(below.result.current.hasReachedViewLimit).toBe(false);

      const atLimit = renderHook(() => useCustomViewDefinitionState(MAX_VIEWS, true, noopPersistView, true));
      expect(atLimit.result.current.hasReachedViewLimit).toBe(true);
    });

    it('does not count an unsaved draft as a persisted view', () => {
      const belowLimit = MAX_VIEWS.slice(0, MAX_CUSTOM_VIEWS_PER_EXPERIMENT - 1);
      const { result } = renderHook(() => useCustomViewDefinitionState(belowLimit, true, noopPersistView, true));

      act(() => result.current.upsertViewContent(makeView('unsaved')));

      expect(result.current.views).toHaveLength(MAX_CUSTOM_VIEWS_PER_EXPERIMENT);
      expect(result.current.hasReachedViewLimit).toBe(false);
    });

    it('blocks starting a new draft at the limit', () => {
      const { result } = renderHook(() => useCustomViewDefinitionState(MAX_VIEWS, true, noopPersistView, true));

      act(() => result.current.startNewView('Blocked draft'));

      expect(result.current.isDraft).toBe(false);
      expect(result.current.draftName).toBe('');
    });

    it('still allows saving an existing view at the limit', async () => {
      const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
      const { result } = renderHook(() => useCustomViewDefinitionState(MAX_VIEWS, true, onPersistView, true));

      act(() => result.current.selectView('view-0'));
      act(() => result.current.upsertViewContent(makeView('view-0', { label: 'edited' })));
      await act(async () => {
        await result.current.saveActiveView();
      });

      expect(onPersistView).toHaveBeenCalledWith(expect.objectContaining({ id: 'view-0', label: 'edited' }));
    });

    it('frees a slot after deleting a persisted view', async () => {
      const onDeleteView = jest.fn<(id: string) => Promise<void>>().mockResolvedValue(undefined);
      const { result } = renderHook(() =>
        useCustomViewDefinitionState(MAX_VIEWS, true, noopPersistView, true, onDeleteView),
      );

      await act(async () => {
        await result.current.deleteView('view-0');
      });

      expect(result.current.hasReachedViewLimit).toBe(false);
      act(() => result.current.startNewView('Now allowed'));
      expect(result.current.isDraft).toBe(true);
    });
  });

  it('does not clobber local edits when a later refetch arrives', () => {
    const first = [makeView('a')];
    const refetched = [makeView('a'), makeView('b')];
    const { result, rerender } = renderHook(
      ({ views, isLoaded }: { views: CustomView[]; isLoaded: boolean }) =>
        useCustomViewDefinitionState(views, isLoaded, noopPersistView, true),
      { initialProps: { views: first, isLoaded: true } },
    );

    expect(result.current.views.map((view) => view.id)).toEqual(['a']);

    // Diverge locally, then a refetch lands: the loaded set must not overwrite
    // the working set.
    act(() => result.current.upsertViewContent(makeView('c')));
    expect(result.current.views.map((view) => view.id)).toEqual(['a', 'c']);

    rerender({ views: refetched, isLoaded: true });
    expect(result.current.views.map((view) => view.id)).toEqual(['a', 'c']);
  });

  describe('autoSelectFirstView', () => {
    it('selects the first persisted view on load', () => {
      const { result } = renderHook(() =>
        useCustomViewDefinitionState(SINGLE_VIEW, true, noopPersistView, true, undefined, true),
      );

      expect(result.current.activeViewId).toBe('a');
      expect(result.current.activeView).toEqual(makeView('a'));
    });

    it('does not select a view when the flag is off', () => {
      const { result } = renderHook(() => useCustomViewDefinitionState(SINGLE_VIEW, true, noopPersistView, true));

      expect(result.current.activeViewId).toBeUndefined();
    });

    it('selects the first view after async load', () => {
      const loaded = [makeView('a'), makeView('b')];
      const { result, rerender } = renderHook(
        ({ views, isLoaded }: { views: CustomView[]; isLoaded: boolean }) =>
          useCustomViewDefinitionState(views, isLoaded, noopPersistView, true, undefined, true),
        { initialProps: { views: NO_VIEWS, isLoaded: false } },
      );

      expect(result.current.activeViewId).toBeUndefined();

      rerender({ views: loaded, isLoaded: true });
      expect(result.current.activeViewId).toBe('a');
    });

    it('does not steal the empty state after the user starts a new draft', () => {
      const { result } = renderHook(() =>
        useCustomViewDefinitionState(SINGLE_VIEW, true, noopPersistView, true, undefined, true),
      );

      expect(result.current.activeViewId).toBe('a');

      act(() => result.current.startNewView('My new view'));
      expect(result.current.isDraft).toBe(true);
      expect(result.current.activeViewId).toBeUndefined();
    });

    it('does not re-select after the user deletes the auto-selected view', async () => {
      const onDeleteView = jest.fn<(id: string) => Promise<void>>().mockResolvedValue(undefined);
      const initialViews = [makeView('a'), makeView('b')];
      const { result } = renderHook(() =>
        useCustomViewDefinitionState(initialViews, true, noopPersistView, true, onDeleteView, true),
      );

      expect(result.current.activeViewId).toBe('a');

      await act(async () => {
        await result.current.deleteView('a');
      });

      expect(result.current.activeViewId).toBeUndefined();
    });
  });
});
