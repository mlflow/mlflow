import { renderHook } from '@testing-library/react';
import { useBrowserKeyShortcutListener } from './useBrowserKeyShortcutListener';
import userEvent from '@testing-library/user-event';

describe('useBrowserKeyShortcutListener', () => {
  const callback = jest.fn();

  beforeEach(() => {
    callback.mockClear();
  });

  it('listens to a single CTRL/CMD+S key combination', async () => {
    renderHook(() => useBrowserKeyShortcutListener('s', { ctrlOrCmdKey: true }, callback));
    expect(callback).not.toHaveBeenCalled();
    await userEvent.keyboard('{Control>}s{/Control}');
    expect(callback).toHaveBeenCalled();
  });

  it('listens to a single ALT/OPT+S key combination', async () => {
    renderHook(() => useBrowserKeyShortcutListener('s', { altOrOptKey: true }, callback));
    expect(callback).not.toHaveBeenCalled();
    await userEvent.keyboard('{Alt>}s{/Alt}');
    expect(callback).toHaveBeenCalled();
  });

  it('listens to a single SHIFT+S key combination', async () => {
    renderHook(() => useBrowserKeyShortcutListener('s', { shiftKey: true }, callback));
    expect(callback).not.toHaveBeenCalled();
    await userEvent.keyboard('{Shift>}s{/Shift}');
    expect(callback).toHaveBeenCalled();
  });

  it('listens to a complex key combination with two modifiers', async () => {
    renderHook(() => useBrowserKeyShortcutListener('s', { altOrOptKey: true, ctrlOrCmdKey: true }, callback));
    expect(callback).not.toHaveBeenCalled();
    await userEvent.keyboard('{Control>}{Alt>}s{/Control}{/Alt}');
    expect(callback).toHaveBeenCalled();
  });

  it('listens to a complex key combination with three modifiers', async () => {
    renderHook(() =>
      useBrowserKeyShortcutListener('s', { altOrOptKey: true, ctrlOrCmdKey: true, shiftKey: true }, callback),
    );
    await userEvent.keyboard('{Shift>}{Alt>}s{/Alt}{/Shift}');
    expect(callback).not.toHaveBeenCalled();
    await userEvent.keyboard('{Shift>}{Control>}s{/Control}{/Shift}');
    expect(callback).not.toHaveBeenCalled();
    await userEvent.keyboard('{Alt>}{Control>}s{/Control}{/Alt}');
    expect(callback).not.toHaveBeenCalled();
    await userEvent.keyboard('{Shift>}{Control>}{Alt>}s{/Control}{/Alt}{/Shift}');
    expect(callback).toHaveBeenCalled();
  });

  it('listens to a complex key combination with three modifiers but sends incomplete combination', async () => {
    renderHook(() =>
      useBrowserKeyShortcutListener('s', { altOrOptKey: true, ctrlOrCmdKey: true, shiftKey: true }, callback),
    );
    expect(callback).not.toHaveBeenCalled();
    await userEvent.keyboard('{Shift>}{Alt>}s{/Alt}{/Shift}');
    expect(callback).not.toHaveBeenCalled();
  });
});
