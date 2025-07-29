import * as fs from 'fs';
import path, { basename } from 'path';

const PYTHON_API_PATH = 'api_reference/source/python_api/';
const files = fs.readdirSync(PYTHON_API_PATH, { withFileTypes: false });
const fileMap = {};

files.forEach((file) => {
  if (file.startsWith('mlflow') && file.endsWith('.rst')) {
    const filename = basename(file, '.rst');
    // the eventual website path
    fileMap[filename] = 'api_reference/python_api/' + filename + '.html';
  }
});

// manual mapping for auth since it's a special case in the docs hierarchy
fileMap['mlflow.server.auth'] = 'api_reference/auth/python-api.html';
fileMap['mlflow.server.cli'] = 'api_reference/cli.html';
fileMap['mlflow.r'] = 'api_reference/R-api.html';
fileMap['mlflow.java'] = 'api_reference/java_api/index.html';
fileMap['mlflow.python'] = 'api_reference/python_api/index.html';
fileMap['mlflow.rest'] = 'api_reference/rest-api.html';
fileMap['mlflow.typescript'] = 'api_reference/typescript_api/index.html';
fileMap['mlflow.llms.deployments.api'] = 'api_reference/llms/deployments/api.html';

// write filemap to json file
fs.writeFileSync('src/api_modules.json', JSON.stringify(fileMap, null, 2));
