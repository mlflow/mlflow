import { NetworkRequestError } from '@databricks/web-shared/errors';

export async function catchNetworkErrorIfExists(error: any): Promise<void> {
  if (error instanceof NetworkRequestError) {
    const body = await error.response?.json();
    error.message = body?.message;
  }

  throw error;
}
