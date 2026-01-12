const INDEXED_DB_NAME = 'MLflow';
const INDEXED_DB_STORE_NAME = 'IndexedDBStore';

export class IndexedDBStorage {
  private readonly dbName: string;
  private readonly storeName: string;
  private readonly onError?: (reason: unknown) => void;

  private db: IDBDatabase | null = null;
  private cache: Map<string, string> = new Map();
  private dirty: Set<string> = new Set();
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private isReady = false;

  public constructor(onError?: (reason: unknown) => void) {
    this.dbName = INDEXED_DB_NAME;
    this.storeName = INDEXED_DB_STORE_NAME;
    this.onError = onError;
  }

  /** Call once at startup. After this resolves, getItem / setItem are synchronous. */
  public async initialize(): Promise<void> {
    if (this.isReady) {
      return;
    }
    this.db = await this.openDB();
    await this.loadAllIntoCache();
    this.isReady = true;
  }

  public getItem(key: string): string | null {
    try {
      this.ensureReady();
      const v = this.cache.get(key);
      return v === undefined ? null : v;
    } catch (error) {
      this.handleError(error, 'Failed to get indexed db item');
      return null;
    }
  }

  public setItem(key: string, value: string): void {
    try {
      this.ensureReady();
      this.cache.set(key, value);
      this.dirty.add(key);
      this.scheduleFlush();
    } catch (error) {
      this.handleError(error, 'Failed to set indexed db item');
    }
  }

  // TODO implement GC logic for Indexed DB to clean up old experiment preferences
  public removeItem(key: string): void {
    try {
      this.ensureReady();
      this.cache.delete(key);
      this.dirty.add(key);
      this.scheduleFlush();
    } catch (error) {
      this.handleError(error, 'Failed to remove indexed db item');
    }
  }

  public async flush(): Promise<void> {
    this.ensureReady();
    await this.flushInternal();
  }

  private ensureReady(): void {
    if (!this.isReady || !this.db) {
      throw new Error('IndexedDBStorage not initialized. Call await initialize() before use.');
    }
  }

  private openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, 1);

      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName);
        }
      };

      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  private loadAllIntoCache(): Promise<void> {
    if (!this.db) throw new Error('DB not open');

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(this.storeName, 'readonly');
      const store = tx.objectStore(this.storeName);
      const req = store.openCursor();

      req.onsuccess = () => {
        const cursor = req.result;
        if (!cursor) return;
        this.cache.set(String(cursor.key), String(cursor.value));
        cursor.continue();
      };

      req.onerror = () => reject(req.error);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  }

  private scheduleFlush(): void {
    if (this.flushTimer) return;

    this.flushTimer = setTimeout(() => {
      this.flushTimer = null;
      (async () => {
        try {
          await this.flushInternal();
        } catch (error) {
          this.handleError(error, 'Failed to save settings');
        }
      })();
    }, 50);
  }

  private flushInternal(): Promise<void> {
    if (!this.db) throw new Error('DB not open');
    if (this.dirty.size === 0) return Promise.resolve();

    const keys = Array.from(this.dirty);
    this.dirty.clear();

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(this.storeName, 'readwrite');
      const store = tx.objectStore(this.storeName);

      for (const key of keys) {
        if (this.cache.has(key)) {
          store.put(this.cache.get(key)!, key);
        } else {
          store.delete(key);
        }
      }

      tx.oncomplete = () => resolve();
      tx.onerror = tx.onabort = () => {
        keys.forEach((e) => this.dirty.add(e));
        reject(tx.error);
      };
    });
  }

  private handleError(error: any, context: string): void {
    if (this.onError) {
      this.onError(new Error(context + (error ? `: ${error}` : '')));
    }
  }
}
