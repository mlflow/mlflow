import * as fs from "fs";
import path, { basename } from "path";

const PYTHON_API_PATH = "api_reference/source/python_api/";
const files = fs.readdirSync(PYTHON_API_PATH, { withFileTypes: false });
const fileMap = {};

files.forEach((file) => {
  if (file.startsWith("mlflow") && file.endsWith(".rst")) {
    const filename = basename(file, ".rst");
    // the eventual website path
    fileMap[filename] = "api_reference/python_api/" + filename + ".html";
  }
});

// manual mapping for auth since it's a special case in the docs hierarchy
fileMap["mlflow.server.auth"] = "api_reference/auth/python-api.html";

// write filemap to json file
fs.writeFileSync("src/api_modules.json", JSON.stringify(fileMap, null, 2));
