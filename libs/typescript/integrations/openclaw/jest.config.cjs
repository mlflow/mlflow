const path = require('path');

/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/tests'],
  testMatch: ['**/*.test.ts'],
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
        },
      },
    ],
  },
  moduleNameMapper: {
    // OpenClaw supplies the plugin SDK at runtime.
    '^openclaw/plugin-sdk/.*$': '<rootDir>/tests/__mocks__/empty.ts',
    // Resolve ESM-style imports to the TypeScript source in Jest.
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },
};
