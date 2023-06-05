/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { ErrorCodes } from '../constants';

export class ErrorWrapper {
  status: any;
  text: any;
  textJson: any;
  constructor(text: any, status = 500) {
    this.status = status;
    this.text = text;
    if (typeof text === 'object') {
      this.textJson = text;
    } else {
      try {
        this.textJson = JSON.parse(text);
      } catch {
        this.textJson = null;
      }
    }
  }

  getStatus() {
    return this.status;
  }

  getUserVisibleError() {
    return this.textJson || 'INTERNAL_SERVER_ERROR';
  }

  getErrorCode() {
    return this.textJson ? this.textJson.error_code : ErrorCodes.INTERNAL_ERROR;
  }

  getMessageField() {
    return this.textJson ? this.textJson.message : ErrorCodes.INTERNAL_ERROR;
  }

  renderHttpError() {
    if (this.textJson) {
      if (this.textJson.error_code && this.textJson.message) {
        const message = this.textJson.error_code + ': ' + this.textJson.message;
        return this.textJson.stack_trace ? `${message}\n\n${this.textJson.stack_trace}` : message;
      } else {
        return this.textJson.message || 'Request Failed';
      }
    } else {
      // Do our best to clean up and return the error: remove any tags, and reduce duplicate
      // newlines.
      let simplifiedText = this.text.replace(/<[^>]+>/gi, '');
      simplifiedText = simplifiedText.replace(/\n\n+/gi, '\n');
      simplifiedText = simplifiedText.trim();
      return simplifiedText;
    }
  }

  /**
   * Returns true if this instance wraps HTTP client (4XX) error
   */
  is4xxError() {
    const status = parseInt(this.getStatus(), 10);
    return status >= 400 && status <= 499;
  }
}
