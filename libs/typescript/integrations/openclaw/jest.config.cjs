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
  // openclaw/plugin-sdk/* doesn't exist outside the gateway runtime —
  // map to a trivial empty module so service.ts can load.
  moduleNameMapper: {
    '^openclaw/plugin-sdk/.*$': '<rootDir>/tests/__mocks__/empty.ts',
  },
};
