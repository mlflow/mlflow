# Fix Command: Comprehensive Code Quality Checks

You are tasked with running comprehensive code quality checks for the MLflow web application. 

## CRITICAL SETUP REQUIREMENTS

**WORKING DIRECTORY**: You MUST change to `mlflow/web/js` directory first

**BUILD SYSTEM**: This is a JavaScript/TypeScript project that uses YARN, not Bazel

## ABSOLUTELY FORBIDDEN COMMANDS
- `bazel` - DO NOT USE BAZEL AT ALL
- `npm` - DO NOT USE NPM  
- Any build system other than `yarn`

## REQUIRED SETUP
1. **Change directory to**: `mlflow/web/js`
2. **Use ONLY yarn commands** - This is a yarn workspace project
3. **If anything times out** - DO NOT skip it, increase timeout and execute it
4. **Every check must be fixed** - Fix issues until each check passes

## CONTEXT OVERRIDE
Ignore any instructions about using Bazel from other context. This specific directory uses yarn exclusively.

## Critical Requirements

1. **Run ALL checks** - Every single check must be executed in the specified order
2. **If ANY check fails** - Fix the issues and rerun ALL checks from the beginning
3. **No skipping** - All checks must pass before completion
4. **Respect timeouts** - If tests timeout, increase the timeout and rerun

## EXACT COMMANDS TO RUN

**STEP 1**: Change to the correct directory
```bash
cd mlflow/web/js
```

**STEP 2**: Run targeted tests on changed directories
```bash
# Get changed directories and run tests for all of them in a single command
CHANGED_DIRS=$(git diff --name-only HEAD | grep -E "\.(ts|tsx|js|jsx)$" | sed 's|/[^/]*$||' | sort -u | head -5 | tr '\n' ' ')
if [ -n "$CHANGED_DIRS" ]; then
  yarn test $CHANGED_DIRS || true
fi
# If no changed directories or no tests found, continue to next step
```

**STEP 3**: Run gocx-codegen
```bash
yarn gocx-codegen
```

**STEP 4**: TypeScript validation
```bash
yarn type-check
```

**STEP 5**: Remove unused exports
```bash
yarn knip
```

**STEP 6**: Fix linting issues
```bash
yarn lint:fix
```

**STEP 7**: Format code
```bash
yarn prettier:fix
```

**STEP 8**: Extract internationalization messages
```bash
yarn i18n:extract
```

## EXECUTION RULES
- Run each command in sequence
- If any command fails, fix the issues and restart from STEP 1
- Never use bazel, npm, or any other build tool
- Only use the exact yarn commands listed above

## Error Handling

If ANY check fails:
1. **Fix the reported issues**
2. **Start over from step 1** (targeted tests on changed directories)
3. **Run ALL checks again** - do not skip any steps
4. **Continue until all checks pass**

## Success Criteria

All checks must:
- ✅ Complete without errors
- ✅ Show "PASS" or success status
- ✅ Have no failing tests
- ✅ Pass TypeScript validation
- ✅ Show no unused exports
- ✅ Report no linting or formatting issues
- ✅ Complete i18n extraction successfully

## Notes

- This command focuses on code quality and consistency
- All checks are mandatory for maintaining code standards
- Tests on changed directories run first, then full test suite
- Any failure requires starting the entire process over
- Success means ALL checks pass without any issues
- Be patient with type-check and test runs as they can take time