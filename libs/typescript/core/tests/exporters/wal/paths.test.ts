import { mkdtemp, rm } from 'fs/promises';
import { tmpdir } from 'os';
import { join, resolve } from 'path';
import {
  getDaemonLogPath,
  getDeadLetterPath,
  getLockSocketPath,
  getWalDir,
  getWalPath,
} from '../../../src/exporters/wal/paths';

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

  describe('daily rotation of failed.log / daemon.log', () => {
    beforeEach(() => {
      process.env.MLFLOW_WAL_DIR = '/tmp/wal-rotation';
    });

    it('suffixes failed.log with the UTC date of the supplied Date', () => {
      const d = new Date('2026-05-20T12:34:56.000Z');
      expect(getDeadLetterPath(d)).toBe(
        join(resolve('/tmp/wal-rotation'), 'failed.log.2026-05-20'),
      );
    });

    it('suffixes daemon.log with the UTC date of the supplied Date', () => {
      const d = new Date('2026-05-20T12:34:56.000Z');
      expect(getDaemonLogPath(d)).toBe(join(resolve('/tmp/wal-rotation'), 'daemon.log.2026-05-20'));
    });

    it('rolls failed.log to the next day across midnight UTC', () => {
      const lastSecondOfDay = new Date('2026-05-20T23:59:59.999Z');
      const firstInstantOfNextDay = new Date('2026-05-21T00:00:00.000Z');
      expect(getDeadLetterPath(lastSecondOfDay)).toMatch(/failed\.log\.2026-05-20$/);
      expect(getDeadLetterPath(firstInstantOfNextDay)).toMatch(/failed\.log\.2026-05-21$/);
    });

    it('rolls daemon.log to the next day across midnight UTC', () => {
      const lastSecondOfDay = new Date('2026-05-20T23:59:59.999Z');
      const firstInstantOfNextDay = new Date('2026-05-21T00:00:00.000Z');
      expect(getDaemonLogPath(lastSecondOfDay)).toMatch(/daemon\.log\.2026-05-20$/);
      expect(getDaemonLogPath(firstInstantOfNextDay)).toMatch(/daemon\.log\.2026-05-21$/);
    });

    it('uses UTC date, not local date, regardless of the process timezone', () => {
      // 2026-05-20T18:00:00 UTC = 2026-05-21T02:00:00 in UTC+8 (e.g. Singapore).
      // The suffix must reflect UTC, not the host TZ.
      const d = new Date('2026-05-20T18:00:00.000Z');
      expect(getDaemonLogPath(d)).toMatch(/daemon\.log\.2026-05-20$/);
      expect(getDeadLetterPath(d)).toMatch(/failed\.log\.2026-05-20$/);
    });

    it('zero-pads month and day', () => {
      const earlyDay = new Date('2026-01-03T12:00:00.000Z');
      expect(getDaemonLogPath(earlyDay)).toMatch(/daemon\.log\.2026-01-03$/);
    });

    it('defaults to "now" when called without an argument', () => {
      expect(getDaemonLogPath()).toMatch(/daemon\.log\.\d{4}-\d{2}-\d{2}$/);
      expect(getDeadLetterPath()).toMatch(/failed\.log\.\d{4}-\d{2}-\d{2}$/);
    });
  });
});
