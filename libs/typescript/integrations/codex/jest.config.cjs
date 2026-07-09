const path = require('path');

/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/tests'],
  testMatch: ['**/*.test.ts'],
  moduleFileExtensions: ['ts', 'js', 'json', 'node'],
  modulePaths: [path.resolve(__dirname, '../../node_modules')],
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        tsconfig: {
          target: 'ES2022',
          module: 'CommonJS',
          moduleResolution: 'Node',
          esModuleInterop: true,
          strict: true,
          skipLibCheck: true,
          types: ['jest', 'node'],
          baseUrl: '.',
          paths: {
            '@mlflow/core': ['../../core/src/index.ts'],
            '@mlflow/core/*': ['../../core/src/*'],
          },
        },
      },
    ],
  },
  moduleNameMapper: {
    '^@mlflow/core$': '<rootDir>/../../core/src',
    '^@mlflow/core/(.*)$': '<rootDir>/../../core/src/$1',
    // Strip .js extensions for ESM → CJS test resolution
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },
  testTimeout: 30000,
  forceExit: true,
  detectOpenHandles: true,
};
