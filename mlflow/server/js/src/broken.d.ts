/**
 * A temp workaround for TS error when exporting withRouter:
 *
 * export const Foo = withRouter(
 *
 * The inferred type of 'Foo' cannot be named without a reference to 'react-router-dom/node_modules/@types/react-router'.
 * This is likely not portable. A type annotation is necessary.ts(2742)
 *
 * Will fix after TS migration.
 */
declare type TODOBrokenReactRouterType = any;

declare module 'cookie';
declare module 'json-bigint';
declare module 'js-yaml';
declare module 'sanitize-html';
declare module 'enzyme';
declare module 'redux-promise-middleware';
declare module 'redux-mock-store';
declare module 'leaflet';
