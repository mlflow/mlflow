/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/tests', '<rootDir>/src'],
  testMatch: ['**/*.test.ts'],
  moduleFileExtensions: ['ts', 'js', 'json', 'node'],
  transform: {
    '^.+\\.tsx?$': ['ts-jest', { tsconfig: 'tsconfig.json' }]
  },
  globalSetup: '<rootDir>/../../jest.global-server-setup.ts',
  globalTeardown: '<rootDir>/../../jest.global-server-teardown.ts',
  testTimeout: 30000,
  forceExit: true,
  detectOpenHandles: true
};
