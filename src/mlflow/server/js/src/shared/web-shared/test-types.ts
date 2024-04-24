type WebSharedDesignSystem = typeof import('@databricks/web-shared/design-system');
type StubDesignSystem = typeof import('./design-system');

type Equal<U1, U2> = [U1] extends [U2] ? ([U2] extends [U1] ? true : false) : false;

// To see the type mismatch, uncomment the following line and adjust the types
// const a: WebSharedDesignSystem = {} as StubDesignSystem;

// All the Equal results should be true
const _testUtilsEqual: [Equal<WebSharedDesignSystem, StubDesignSystem>] = [true];
