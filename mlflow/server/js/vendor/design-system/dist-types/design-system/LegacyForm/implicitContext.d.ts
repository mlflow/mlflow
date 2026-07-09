type AnyFunction = (...args: any[]) => any;
export interface ImplicitContextValue {
    wrap: <T extends AnyFunction>(callback: T) => T;
    wrapNoProxy: <T extends AnyFunction>(callback: T) => T;
}
export type ImplicitContextGetter = () => ImplicitContextValue;
export declare function setImplicitContextGetter(getter: ImplicitContextGetter): void;
export declare function getImplicitContext(): ImplicitContextValue;
export declare function resetImplicitContextGetterForTests(): void;
export {};
//# sourceMappingURL=implicitContext.d.ts.map