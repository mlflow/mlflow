import { ErrorCodes } from '../../common/constants';

export const isPendingApi = (action) => {
  return action.type.endsWith('_PENDING');
};

export const pending = (apiActionType) => {
  return `${apiActionType}_PENDING`;
};

export const isFulfilledApi = (action) => {
  return action.type.endsWith('_FULFILLED');
};

export const fulfilled = (apiActionType) => {
  return `${apiActionType}_FULFILLED`;
};

export const isRejectedApi = (action) => {
  return action.type.endsWith('_REJECTED');
};

export const rejected = (apiActionType) => {
  return `${apiActionType}_REJECTED`;
};
/**
 * Wraps a Jquery AJAX request (passed via `deferred`) in a new Promise which resolves and
 * rejects using the ajax callbacks `success` and `error`. Retries with exponential backoff
 * if the server responds with a 429 (Too Many Requests).
 * @param {function} deferred - Function with signature ({data, success, error}) => Any, where
 *   data is a JSON payload for an AJAX request, success is a callback to execute on request
 *   success, and error is a callback to execute on request failure.
 * @param {object} data - Data argument to pass to `deferred`
 * @param {int} timeLeftMs - Time left to retry the AJAX request in ms, if we receive a 429
 * response from the server. Defaults to 60 seconds.
 * @param {int} sleepMs - Time to sleep before retrying the AJAX request if we receive a 429
 * response from the server. Defaults to 1 second.
 */
export const wrapDeferred = (deferred, data, timeLeftMs = 60000, sleepMs = 1000) => {
  return new Promise((resolve, reject) => {
    deferred({
      data,
      success: (response) => {
        resolve(response);
      },
      error: (xhr) => {
        if (xhr.status === 429) {
          if (timeLeftMs > 0) {
            console.warn(
              'Request failed with status code 429, message ' +
                new ErrorWrapper(xhr).getUserVisibleError() +
                '. Retrying after ' +
                sleepMs +
                ' ms. On additional 429 errors, will continue to retry for up ' +
                'to ' +
                timeLeftMs +
                ' ms.',
            );
            // Retry the request, subtracting the current sleep duration from the remaining time
            // and doubling the sleep duration
            const newTimeLeft = timeLeftMs - sleepMs;
            const newSleepMs = Math.min(newTimeLeft, sleepMs * 2);
            return new Promise((resolveRetry) => setTimeout(resolveRetry, sleepMs))
              .then(() => {
                return wrapDeferred(deferred, data, newTimeLeft, newSleepMs);
              })
              .then(
                (successResponse) => resolve(successResponse),
                (failureResponse) => reject(failureResponse),
              );
          }
        }
        console.error('XHR failed', xhr);
        // We can't throw the XHR itself because it looks like a promise to the
        // redux-promise-middleware.
        return reject(new ErrorWrapper(xhr));
      },
    });
  });
};

export class ErrorWrapper {
  constructor(xhr) {
    this.xhr = xhr;
  }

  getErrorCode() {
    const { responseText } = this.xhr;
    if (responseText) {
      try {
        const parsed = JSON.parse(responseText);
        if (parsed.error_code) {
          return parsed.error_code;
        }
      } catch (e) {
        return ErrorCodes.INTERNAL_ERROR;
      }
    }
    return ErrorCodes.INTERNAL_ERROR;
  }

  // Return the responseText if it is in the
  // { error_code: ..., message: ...} format. Otherwise return "INTERNAL_SERVER_ERROR".
  getUserVisibleError() {
    const { responseText } = this.xhr;
    if (responseText) {
      try {
        const parsed = JSON.parse(responseText);
        if (parsed.error_code) {
          return responseText;
        }
      } catch (e) {
        return 'INTERNAL_SERVER_ERROR';
      }
    }
    return 'INTERNAL_SERVER_ERROR';
  }

  getMessageField() {
    const { responseText } = this.xhr;
    if (responseText) {
      try {
        const parsed = JSON.parse(responseText);
        if (parsed.error_code && parsed.message) {
          return parsed.message;
        }
      } catch (e) {
        return 'INTERNAL_SERVER_ERROR';
      }
    }
    return 'INTERNAL_SERVER_ERROR';
  }

  getStatus() {
    const { status } = this.xhr;
    return status;
  }

  // Does a best-effort at rendering an arbitrary HTTP error. If it's a DatabricksServiceException,
  // will render the error code and message. If it's HTML, we'll strip the tags.
  renderHttpError() {
    const { responseText } = this.xhr;
    if (responseText) {
      try {
        const parsed = JSON.parse(responseText);
        if (parsed.error_code && parsed.message) {
          const message = parsed.error_code + ': ' + parsed.message;
          if (parsed.stack_trace) {
            return message + '\n\n' + parsed.stack_trace;
          } else {
            return message;
          }
        }
      } catch (e) {
        // Do our best to clean up and return the error: remove any tags, and reduce duplicate
        // newlines.
        let simplifiedText = responseText.replace(/<[^>]+>/gi, '');
        simplifiedText = simplifiedText.replace(/\n\n+/gi, '\n');
        simplifiedText = simplifiedText.trim();
        return simplifiedText;
      }
    }
    return 'Request Failed';
  }
}

export const getUUID = () => {
  const randomPart = Math.random()
    .toString(36)
    .substring(2, 10);
  return new Date().getTime() + randomPart;
};
