import { ensureInitialized } from '../src/config';
import { runNotifyHook } from '../src/hooks/stop';
import { processNotify } from '../src/tracing';

jest.mock('../src/config', () => ({ ensureInitialized: jest.fn() }));
jest.mock('../src/tracing', () => ({ processNotify: jest.fn() }));

const ensureInitializedMock = jest.mocked(ensureInitialized);
const processNotifyMock = jest.mocked(processNotify);

describe('runNotifyHook', () => {
  let consoleErrorSpy: jest.SpyInstance;

  beforeEach(() => {
    ensureInitializedMock.mockReturnValue(true);
    processNotifyMock.mockReset();
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    consoleErrorSpy.mockRestore();
  });

  it('logs the error stack without credential-bearing properties', async () => {
    const error = Object.assign(new Error('authentication failed'), {
      config: { token: 'secret-token' },
    });
    processNotifyMock.mockRejectedValue(error);

    await runNotifyHook(JSON.stringify({ type: 'agent-turn-complete' }));

    expect(consoleErrorSpy).toHaveBeenCalledWith('[mlflow]', error.stack);
    expect(JSON.stringify(consoleErrorSpy.mock.calls)).not.toContain('secret-token');
  });

  it('logs a string representation when a non-Error value is thrown', async () => {
    processNotifyMock.mockRejectedValue('notify failed');

    await runNotifyHook(JSON.stringify({ type: 'agent-turn-complete' }));

    expect(consoleErrorSpy).toHaveBeenCalledWith('[mlflow]', 'notify failed');
  });
});
