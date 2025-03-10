import type { ErrorLogType } from './ErrorLogType';
import type { ErrorName } from './ErrorName';
import {
  PredefinedError,
  matchPredefinedError,
  NotFoundError,
  PermissionError,
  UnauthorizedError,
  UnknownError,
} from './PredefinedErrors';

class GenericError extends PredefinedError {
  errorLogType = 'GenericError' as ErrorLogType;
  errorName = 'GenericError' as ErrorName;
  displayMessage = 'Generic Message';
}

describe('PredefinedErrors', () => {
  describe('matchPredefinedError', () => {
    it('should properly handle errors which are Responses', () => {
      const testResponseError = new Response(null, { status: 404 });

      const matchedError = matchPredefinedError(testResponseError);

      expect(matchedError).toBeInstanceOf(NotFoundError);
    });

    it('should properly handle errors which are Apollo Errors', () => {
      const testApolloError = new Error('Test Apollo Error');
      const networkError = new Error();
      Object.assign(networkError, { statusCode: 401, response: new Response(null, { status: 401 }) });
      (testApolloError as any).networkError = networkError;
      const matchedError = matchPredefinedError(testApolloError);

      expect(matchedError).toBeInstanceOf(UnauthorizedError);
    });

    it('should pass through predefined error', () => {
      const testPredefinedError = new PermissionError({});
      const matchedError = matchPredefinedError(testPredefinedError);

      expect(matchedError).toEqual(testPredefinedError);
    });

    it('should pass through unknown error if unable to match', () => {
      const testUnknownError = new Error('Test Unknown Error');
      const matchedError = matchPredefinedError(testUnknownError);

      expect(matchedError).toBeInstanceOf(UnknownError);
    });
  });
});
