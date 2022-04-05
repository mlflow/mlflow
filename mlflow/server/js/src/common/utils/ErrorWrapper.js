import { ErrorCodes } from '../constants';

export class ErrorWrapper {
  constructor(text, status = 500) {
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
}
