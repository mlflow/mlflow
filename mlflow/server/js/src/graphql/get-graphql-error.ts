import { ApolloError } from '@apollo/client';

interface CommonGraphQLApiError {
  code: string | null;
  message: string | null;
}

export const getGraphQLErrorMessage = (error?: CommonGraphQLApiError | ApolloError | Error | any) => {
  if (!error) {
    return undefined;
  }
  if (error instanceof ApolloError) {
    if (error.graphQLErrors.length > 0) {
      return error.graphQLErrors.map((e) => e.toString()).join(', ');
    }
  }

  if ('message' in error) {
    return error.message;
  }

  return error.toString();
};
