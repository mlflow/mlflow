import * as v1 from '../src/v1';

describe('OpenCode V1 entry point', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('exports only plugin functions because OpenCode V1 calls every export', () => {
    for (const value of Object.values(v1)) {
      expect(typeof value).toBe('function');
    }
    expect(v1.default).toBe(v1.MLflowTracingPlugin);
  });

  it('tells V1 users to pin 0.4 and registers no hooks', async () => {
    const warning = jest.spyOn(console, 'error').mockImplementation(() => {});
    const showToast = jest.fn().mockResolvedValue(true);
    const hooks = await v1.MLflowTracingPlugin({ client: { tui: { showToast } } });
    expect(hooks).toEqual({});
    expect(warning).toHaveBeenCalledWith(expect.stringContaining('@mlflow/opencode@0.4'));
    expect(showToast).toHaveBeenCalledWith({
      body: expect.objectContaining({
        variant: 'warning',
        message: expect.stringContaining('@mlflow/opencode@0.4'),
      }),
    });
  });

  it('does not fail when the TUI is unavailable', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    const showToast = jest.fn().mockRejectedValue(new Error('No TUI'));
    await expect(v1.MLflowTracingPlugin({ client: { tui: { showToast } } })).resolves.toEqual({});
    await expect(v1.MLflowTracingPlugin()).resolves.toEqual({});
  });
});
