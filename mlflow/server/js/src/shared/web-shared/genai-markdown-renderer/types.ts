import type { ComponentProps, ComponentType } from 'react';
import type { ExtraProps } from 'react-markdown-10';

/**
 * The exported type for Components in react-markdown-10 provides "any" as the type for the props passed to components,
 * and also doesn't doesn't have type safety for the tag names.
 * This type is a workaround to provide type safety for the tag names and the props passed to components.
 */
export type ReactMarkdownComponents = {
  [K in keyof JSX.IntrinsicElements]?: ReactMarkdownComponent<K>;
};

export type ReactMarkdownComponent<T extends keyof JSX.IntrinsicElements> = ComponentType<
  React.PropsWithChildren<ReactMarkdownProps<T>>
>;

export type ReactMarkdownProps<T extends keyof JSX.IntrinsicElements> = ComponentProps<T> & ExtraProps;
