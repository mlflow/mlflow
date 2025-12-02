const DEFAULT_SCOPE = 'all-apis';
const TOKEN_REFRESH_SKEW_MS = 60_000;

export interface DatabricksAuthProviderOptions {
  host: string;
  initialToken?: string;
  clientId?: string;
  clientSecret?: string;
  scopes?: string[];
  tokenEndpoint?: string;
  fetchFn?: typeof fetch;
}

export class DatabricksAuthProvider {
  private readonly host: string;
  private readonly staticToken?: string;
  private readonly clientId?: string;
  private readonly clientSecret?: string;
  private readonly scopes: string[];
  private readonly tokenEndpoint?: string;
  private readonly fetchFn: typeof fetch;

  private cachedToken?: string;
  private tokenExpiryMs?: number;
  private ongoingRefresh?: Promise<void>;

  constructor(options: DatabricksAuthProviderOptions) {
    this.host = options.host;
    this.staticToken = options.initialToken;
    this.clientId = options.clientId;
    this.clientSecret = options.clientSecret;
    this.scopes = options.scopes && options.scopes.length > 0 ? options.scopes : [DEFAULT_SCOPE];
    this.tokenEndpoint = options.tokenEndpoint;
    this.fetchFn = options.fetchFn ?? fetch;

    if (this.staticToken) {
      this.cachedToken = this.staticToken;
      this.tokenExpiryMs = Number.POSITIVE_INFINITY;
    }
  }

  async getAccessToken(): Promise<string | undefined> {
    if (this.staticToken) {
      return this.staticToken;
    }

    if (!this.clientId || !this.clientSecret) {
      return undefined;
    }

    if (this.cachedToken && this.tokenExpiryMs && Date.now() < this.tokenExpiryMs - TOKEN_REFRESH_SKEW_MS) {
      return this.cachedToken;
    }

    if (!this.ongoingRefresh) {
      this.ongoingRefresh = this.refreshAccessToken();
    }

    try {
      await this.ongoingRefresh;
    } finally {
      this.ongoingRefresh = undefined;
    }

    return this.cachedToken;
  }

  private async refreshAccessToken(): Promise<void> {
    const endpoint = this.tokenEndpoint ?? new URL('/oidc/token', this.host).toString();
    const body = new URLSearchParams({
      grant_type: 'client_credentials',
      scope: this.scopes.join(' ')
    }).toString();

    const credentials = encodeBase64(`${this.clientId}:${this.clientSecret}`);
    const response = await this.fetchFn(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        Authorization: `Basic ${credentials}`
      },
      body
    });

    if (!response.ok) {
      const errorText = await safeReadText(response);
      throw new Error(
        `Failed to exchange Databricks client credentials: ${response.status} ${response.statusText}${errorText ? ` - ${errorText}` : ''}`
      );
    }

    const payload = (await response.json()) as {
      access_token?: string;
      expires_in?: number;
    };

    if (!payload.access_token) {
      throw new Error('Databricks token response missing access_token');
    }

    this.cachedToken = payload.access_token;
    const expiresInSec = typeof payload.expires_in === 'number' ? payload.expires_in : 3600;
    this.tokenExpiryMs = Date.now() + expiresInSec * 1000;
  }
}

async function safeReadText(response: Response): Promise<string> {
  try {
    return await response.text();
  } catch {
    return '';
  }
}

function encodeBase64(value: string): string {
  const globalBuffer = (globalThis as {
    Buffer?: { from(input: string): { toString(encoding: string): string } };
  }).Buffer;
  if (globalBuffer) {
    return globalBuffer.from(value).toString('base64');
  }

  const browserBtoa = (globalThis as { btoa?: (data: string) => string }).btoa;
  if (browserBtoa) {
    return browserBtoa(value);
  }

  throw new Error('Base64 encoding is not supported in this environment');
}
