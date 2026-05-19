import { mkdtemp, rm } from 'fs/promises';
import { tmpdir } from 'os';
import { join, resolve } from 'path';
import { getLockSocketPath, getWalDir, getWalPath } from '../../../src/exporters/wal/paths';

describe('wal/paths', () => {
  const originalEnv = process.env.MLFLOW_WAL_DIR;

  afterEach(() => {
    if (originalEnv === undefined) {
      delete process.env.MLFLOW_WAL_DIR;
    } else {
      process.env.MLFLOW_WAL_DIR = originalEnv;
    }
  });

  describe('getWalDir', () => {
    it('honors MLFLOW_WAL_DIR and returns an absolute, resolved path', () => {
      process.env.MLFLOW_WAL_DIR = '/tmp/foo/';
      expect(getWalDir()).toBe(resolve('/tmp/foo'));
    });

    it('treats empty MLFLOW_WAL_DIR as unset', () => {
      process.env.MLFLOW_WAL_DIR = '';
      expect(getWalDir()).toMatch(/[/\\]\.mlflow[/\\]wal$/);
    });

    it('produces the same value for /tmp/foo and /tmp/foo/', () => {
      process.env.MLFLOW_WAL_DIR = '/tmp/foo';
      const a = getWalDir();
      process.env.MLFLOW_WAL_DIR = '/tmp/foo/';
      const b = getWalDir();
      expect(a).toBe(b);
    });
  });

  describe('getLockSocketPath', () => {
    let scratchDir: string;

    beforeEach(async () => {
      scratchDir = await mkdtemp(join(tmpdir(), 'mlflow-paths-'));
      process.env.MLFLOW_WAL_DIR = scratchDir;
    });

    afterEach(async () => {
      await rm(scratchDir, { recursive: true, force: true });
    });

    const isWindows = process.platform === 'win32';

    (isWindows ? it.skip : it)('POSIX: returns <wal_dir>/daemon.sock for short WAL dirs', () => {
      const lock = getLockSocketPath();
      expect(lock).toBe(join(scratchDir, 'daemon.sock'));
    });

    (isWindows ? it.skip : it)(
      'POSIX: falls back to tmpdir-based path when wal_dir would exceed sun_path',
      () => {
        // Construct a long WAL dir that pushes <wal_dir>/daemon.sock past 103
        // bytes. We do not need this directory to exist; getLockSocketPath
        // only inspects path lengths.
        const longWalDir = '/tmp/' + 'x'.repeat(150);
        process.env.MLFLOW_WAL_DIR = longWalDir;

        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        try {
          const lock = getLockSocketPath();
          expect(lock.startsWith(tmpdir())).toBe(true);
          expect(lock).toMatch(/[/\\]mlflow-[A-Za-z0-9._-]+-[0-9a-f]{12}\.sock$/);
          expect(Buffer.byteLength(lock, 'utf8')).toBeLessThanOrEqual(103);
          expect(warnSpy).toHaveBeenCalledTimes(1);
          expect(warnSpy.mock.calls[0]?.[0]).toContain('longer than 103 bytes');
        } finally {
          warnSpy.mockRestore();
        }
      },
    );

    (isWindows ? it.skip : it)(
      'POSIX: fallback identifier is stable for the same wal_dir and differs for distinct wal_dirs',
      () => {
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        try {
          process.env.MLFLOW_WAL_DIR = '/tmp/' + 'a'.repeat(150);
          const lockA1 = getLockSocketPath();
          const lockA2 = getLockSocketPath();
          process.env.MLFLOW_WAL_DIR = '/tmp/' + 'b'.repeat(150);
          const lockB = getLockSocketPath();
          expect(lockA1).toBe(lockA2);
          expect(lockA1).not.toBe(lockB);
        } finally {
          warnSpy.mockRestore();
        }
      },
    );

    (isWindows ? it : it.skip)(
      'Windows: returns a Named Pipe path scoped by user + wal_dir hash',
      () => {
        const lock = getLockSocketPath();
        expect(lock).toMatch(/^\\\\\?\\pipe\\mlflow-trace-daemon-[A-Za-z0-9._-]+-[0-9a-f]{12}$/);
      },
    );

    (isWindows ? it : it.skip)('Windows: pipe name differs for distinct wal_dirs', () => {
      process.env.MLFLOW_WAL_DIR = 'C:\\tmp\\walA';
      const lockA = getLockSocketPath();
      process.env.MLFLOW_WAL_DIR = 'C:\\tmp\\walB';
      const lockB = getLockSocketPath();
      expect(lockA).not.toBe(lockB);
    });
  });

  describe('getWalPath', () => {
    it('lives inside the resolved WAL dir', () => {
      process.env.MLFLOW_WAL_DIR = '/tmp/foo';
      expect(getWalPath()).toBe(join(resolve('/tmp/foo'), 'queue.log'));
    });
  });
});
