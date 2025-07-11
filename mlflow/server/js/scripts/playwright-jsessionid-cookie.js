#!/usr/bin/env node

/**
 * MLflow Playwright JSESSIONID Setup
 *
 * PURPOSE:
 * Sets up Playwright browser automation with automatic JSESSIONID extraction from Chrome
 * for MLflow UI testing and automation. This script enables Claude to interact with the
 * MLflow web interface by automatically extracting your Chrome session and configuring
 * Playwright with the proper authentication.
 *
 * USAGE:
 *
 * Via yarn scripts (recommended):
 *   yarn playwright-jsessionid-cookie refresh    # Setup/refresh automation
 *   yarn playwright-jsessionid-cookie status     # Check current setup
 *
 * WHAT IT DOES:
 * 1. Extracts JSESSIONID from your Chrome browser for dev.local:22090
 * 2. Creates a Playwright storage file with the session cookie
 * 3. Configures Playwright MCP server in Claude with user scope
 * 4. Enables Claude to bypass login screens when automating MLflow UI
 *
 * PREREQUISITES:
 * - Login to https://dev.local:22090 in Chrome first
 * - MLflow dev server running (from universe-claude root)
 * - Restart Claude Code after running this script
 *
 * COMMANDS:
 * - refresh: Extract new JSESSIONID and update Playwright configuration
 * - status:  Check current setup and configuration status
 *
 * OPTIONS:
 * --profile: Chrome profile name (default: 'Default')
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync } = require('child_process');
const { ArgumentParser } = require('argparse');

// Configuration
const DOMAIN = 'dev.local';
const ORIGIN = 'https://dev.local:22090';
const CACHE_DIR = path.join(os.homedir(), '.cache', 'claude-mlflow');
const STORAGE_FILE = path.join(CACHE_DIR, 'storage.json');
const MCP_CONFIG_FILE = path.join(os.homedir(), '.config', 'claude', 'mcp', 'servers.json');

// Colors for console output
const colors = {
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  reset: '\x1b[0m',
  bold: '\x1b[1m',
};

function log(message, color = colors.reset) {
  process.stdout.write(`${color}${message}${colors.reset}\n`);
}

function logSuccess(message) {
  log(`✓ ${message}`, colors.green);
}

function logWarning(message) {
  log(`⚠ ${message}`, colors.yellow);
}

function logError(message) {
  log(`✗ ${message}`, colors.red);
}

function logStep(step, message) {
  log(`\n${colors.yellow}Step ${step}: ${message}${colors.reset}`);
}

/**
 * Extract JSESSIONID from Chrome cookies using the Python getcookie script
 */
function extractJSessionId(profile = 'Default') {
  const pythonScriptPath = path.join(
    __dirname,
    '..',
    '..',
    '..',
    '..',
    'managed-evals',
    'ai-tools',
    'ui',
    'getcookie.py',
  );

  if (!fs.existsSync(pythonScriptPath)) {
    throw new Error(`Python getcookie script not found at: ${pythonScriptPath}`);
  }

  try {
    const result = execSync(`python3 "${pythonScriptPath}" "${DOMAIN}" "${profile}"`, {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    const jsessionid = result.trim();
    if (!jsessionid) {
      throw new Error(`No JSESSIONID cookie found for domain: ${DOMAIN}`);
    }

    return jsessionid;
  } catch (error) {
    if (error.stderr) {
      throw new Error(`Cookie extraction failed: ${error.stderr.trim()}`);
    }
    throw new Error(`Cookie extraction failed: ${error.message}`);
  }
}

/**
 * Create Playwright storage file with JSESSIONID
 */
function createStorageFile(jsessionid) {
  // Ensure cache directory exists
  if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
  }

  const storageData = {
    cookies: [
      {
        name: 'JSESSIONID',
        value: jsessionid,
        domain: DOMAIN,
        path: '/',
      },
    ],
    origins: [
      {
        origin: ORIGIN,
        localStorage: [],
        sessionStorage: [],
      },
    ],
  };

  fs.writeFileSync(STORAGE_FILE, JSON.stringify(storageData, null, 2));
  logSuccess(`Created storage.json with JSESSIONID`);
}

/**
 * Check if claude CLI is available
 */
function isClaudeCLIAvailable() {
  try {
    execSync('claude --version', { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

/**
 * Add Playwright MCP server manually
 */
function addMCPServerManually() {
  const configDir = path.dirname(MCP_CONFIG_FILE);

  // Ensure config directory exists
  if (!fs.existsSync(configDir)) {
    fs.mkdirSync(configDir, { recursive: true });
  }

  let config = { mcpServers: {} };

  // Load existing config if it exists
  if (fs.existsSync(MCP_CONFIG_FILE)) {
    // Create backup
    const backupFile = `${MCP_CONFIG_FILE}.backup.${Date.now()}`;
    fs.copyFileSync(MCP_CONFIG_FILE, backupFile);

    try {
      config = JSON.parse(fs.readFileSync(MCP_CONFIG_FILE, 'utf8'));
    } catch (error) {
      logWarning(`Could not parse existing config, creating new one: ${error.message}`);
    }
  }

  // Add Playwright server config
  config.mcpServers.playwright = {
    command: 'npx',
    args: ['@playwright/mcp@latest', '--isolated', `--storage-state=${STORAGE_FILE}`],
  };

  fs.writeFileSync(MCP_CONFIG_FILE, JSON.stringify(config, null, 2));
  logSuccess('Added Playwright MCP server to user configuration');
  log(`Config location: ${MCP_CONFIG_FILE}`, colors.yellow);
}

/**
 * Refresh command - update JSESSIONID and ensure MCP server is configured
 */
function refresh(args) {
  log('Setting up/refreshing MLflow Playwright automation...');

  // Step 1: Extract and save JSESSIONID
  logStep(1, 'Extracting JSESSIONID from Chrome');
  let jsessionid;
  try {
    jsessionid = extractJSessionId(args.profile);
    if (!jsessionid) {
      throw new Error('Empty JSESSIONID retrieved');
    }
    logSuccess('JSESSIONID extracted successfully');
  } catch (error) {
    logError(`Failed to extract JSESSIONID: ${error.message}`);
    logError('Make sure you are logged into https://dev.local:22090 in Chrome');
    process.exit(1);
  }

  createStorageFile(jsessionid);

  // Step 2: Ensure MCP server is configured
  logStep(2, 'Configuring Playwright MCP server');

  if (isClaudeCLIAvailable()) {
    log('Using Claude CLI to configure MCP server...');

    try {
      // Remove any existing playwright server
      execSync('claude mcp remove playwright -s user', { stdio: 'ignore' });
    } catch {
      // Ignore errors if server doesn't exist
    }

    // Add the playwright server with user scope and storage state
    try {
      execSync(
        `claude mcp add -s user playwright npx -- @playwright/mcp@latest --isolated --storage-state="${STORAGE_FILE}"`,
        {
          stdio: 'inherit',
        },
      );
      logSuccess('Playwright MCP server configured with user scope');
    } catch (error) {
      logError(`Failed to configure MCP server: ${error.message}`);
      process.exit(1);
    }
  } else {
    logWarning('Claude CLI not found. Setting up manual configuration...');
    addMCPServerManually();
  }

  // Step 3: Done
  log('\n' + colors.green + 'Setup/refresh complete!' + colors.reset);
  log('='.repeat(24));
  log('');
  log('Next step: Restart Claude Code for changes to take effect');
  log('');
  log('After restart, Playwright automation will be ready for MLflow UI');
}

/**
 * Status command - check current setup
 */
function status() {
  log('MLflow Playwright automation status:');
  log('='.repeat(36));

  // Check storage file
  if (fs.existsSync(STORAGE_FILE)) {
    logSuccess(`Storage file exists: ${STORAGE_FILE}`);
    try {
      const storage = JSON.parse(fs.readFileSync(STORAGE_FILE, 'utf8'));
      const jsessionid = storage.cookies?.find((c) => c.name === 'JSESSIONID')?.value;
      if (jsessionid) {
        log(`  JSESSIONID: ${jsessionid.substring(0, 8)}...${jsessionid.substring(jsessionid.length - 8)}`);
      }
    } catch (error) {
      logWarning(`Could not parse storage file: ${error.message}`);
    }
  } else {
    logError(`Storage file not found: ${STORAGE_FILE}`);
  }

  // Check MCP config
  if (fs.existsSync(MCP_CONFIG_FILE)) {
    logSuccess(`MCP config exists: ${MCP_CONFIG_FILE}`);
    try {
      const config = JSON.parse(fs.readFileSync(MCP_CONFIG_FILE, 'utf8'));
      if (config.mcpServers?.playwright) {
        logSuccess('Playwright MCP server is configured');
      } else {
        logWarning('Playwright MCP server not found in config');
      }
    } catch (error) {
      logWarning(`Could not parse MCP config: ${error.message}`);
    }
  } else {
    logError(`MCP config not found: ${MCP_CONFIG_FILE}`);
  }

  // Check Claude CLI
  if (isClaudeCLIAvailable()) {
    logSuccess('Claude CLI is available');
  } else {
    logWarning('Claude CLI not found');
  }
}

function main() {
  const parser = new ArgumentParser({
    description: 'MLflow Playwright JSESSIONID automation setup',
    add_help: true,
  });

  const subparsers = parser.add_subparsers({
    dest: 'command',
    help: 'Available commands',
  });

  // Refresh command
  const refreshParser = subparsers.add_parser('refresh', {
    help: 'Refresh JSESSIONID when session expires',
  });
  refreshParser.add_argument('--profile', {
    default: 'Default',
    help: 'Chrome profile name (default: Default)',
  });

  // Status command
  subparsers.add_parser('status', {
    help: 'Check current setup status',
  });

  const args = parser.parse_args();

  if (!args.command) {
    parser.print_help();
    process.exit(1);
  }

  try {
    switch (args.command) {
      case 'refresh':
        refresh(args);
        break;
      case 'status':
        status(args);
        break;
      default:
        parser.print_help();
        process.exit(1);
    }
  } catch (error) {
    logError(`Command failed: ${error.message}`);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = {
  extractJSessionId,
  createStorageFile,
  refresh,
  status,
};
