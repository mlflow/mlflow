declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NODE_ENV?: 'development' | 'production' | 'test';
    }
  }
}

// Need dummy export to convince TS that this file is a module
export {};
