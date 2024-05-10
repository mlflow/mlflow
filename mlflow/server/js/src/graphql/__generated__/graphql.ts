export type Maybe<T> = T | null;
export type InputMaybe<T> = Maybe<T>;
export type Exact<T extends { [key: string]: unknown }> = { [K in keyof T]: T[K] };
export type MakeOptional<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]?: Maybe<T[SubKey]> };
export type MakeMaybe<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]: Maybe<T[SubKey]> };
export type Incremental<T> = T | { [P in keyof T]?: P extends ' $fragmentName' | '__typename' ? T[P] : never };
/** All built-in and custom scalars, mapped to their actual values */
export type Scalars = {
  ID: { input: string; output: string; }
  String: { input: string; output: string; }
  Int: { input: number; output: number; }
  Float: { input: number; output: number; }
  /** LongString Scalar type to prevent truncation to max integer in JavaScript. */
  LongString: { input: GraphQLLongString; output: GraphQLLongString; }
};

export type TestQueryVariables = Exact<{ [key: string]: never; }>;


export type TestQuery = { test: { __typename: 'Test', output: string | null } | null };
