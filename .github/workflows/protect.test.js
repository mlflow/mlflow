// Test file for protect.js isNewerRun function

// Mock the isNewerRun function - will be replaced with actual implementation
function isNewerRun(newRun, existingRun) {
  // Returns true if newRun should replace existingRun
  if (!existingRun) return true;

  // If they are different workflow runs, prefer the one created later
  if (newRun.id !== existingRun.id) {
    return new Date(newRun.created_at) > new Date(existingRun.created_at);
  }

  // Same workflow run: higher run_attempt takes priority (re-runs)
  return newRun.run_attempt > existingRun.run_attempt;
}

// Test cases
function runTests() {
  console.log("Running isNewerRun tests...\n");

  let passed = 0;
  let failed = 0;

  function test(name, actual, expected) {
    if (actual === expected) {
      console.log(`✅ PASS: ${name}`);
      passed++;
    } else {
      console.log(`❌ FAIL: ${name}`);
      console.log(`  Expected: ${expected}`);
      console.log(`  Actual: ${actual}`);
      failed++;
    }
  }

  // Test 1: newRun should replace null/undefined
  test(
    "Should replace null/undefined existingRun",
    isNewerRun({ id: 1, run_attempt: 1, created_at: "2024-01-01T00:00:00Z" }, null),
    true
  );

  // Test 2: Different workflow runs - newer created_at should win
  test(
    "Different runs: newer created_at should replace older (even with higher attempt)",
    isNewerRun(
      { id: 222, run_attempt: 1, created_at: "2024-01-02T00:00:00Z" }, // New run, created later
      { id: 111, run_attempt: 3, created_at: "2024-01-01T00:00:00Z" } // Old run, re-run attempt 3
    ),
    true
  );

  // Test 3: Different workflow runs - older created_at should NOT replace newer
  test(
    "Different runs: older created_at should NOT replace newer",
    isNewerRun(
      { id: 111, run_attempt: 3, created_at: "2024-01-01T00:00:00Z" }, // Old run, re-run attempt 3
      { id: 222, run_attempt: 1, created_at: "2024-01-02T00:00:00Z" } // New run, created later
    ),
    false
  );

  // Test 4: Same workflow run - higher run_attempt should replace lower
  test(
    "Same run: higher run_attempt should replace lower",
    isNewerRun(
      { id: 100, run_attempt: 2, created_at: "2024-01-01T00:00:00Z" },
      { id: 100, run_attempt: 1, created_at: "2024-01-01T00:00:00Z" }
    ),
    true
  );

  // Test 5: Same workflow run - lower run_attempt should NOT replace higher
  test(
    "Same run: lower run_attempt should NOT replace higher",
    isNewerRun(
      { id: 100, run_attempt: 1, created_at: "2024-01-01T00:00:00Z" },
      { id: 100, run_attempt: 2, created_at: "2024-01-01T00:00:00Z" }
    ),
    false
  );

  // Test 6: Same workflow run, same attempt - should not replace
  test(
    "Same run, same attempt: should not replace",
    isNewerRun(
      { id: 100, run_attempt: 1, created_at: "2024-01-01T00:00:00Z" },
      { id: 100, run_attempt: 1, created_at: "2024-01-01T00:00:00Z" }
    ),
    false
  );

  // Test 7: Real-world scenario from PR #20623
  test(
    "Real-world: New successful run (attempt 1) should replace old failed re-run (attempt 3)",
    isNewerRun(
      { id: 12345678, run_attempt: 1, created_at: "2024-02-06T10:00:00Z" }, // New successful run
      { id: 12345670, run_attempt: 3, created_at: "2024-02-06T08:00:00Z" } // Old failed run, re-run
    ),
    true
  );

  console.log(`\n${passed} passed, ${failed} failed`);
  return failed === 0;
}

// Run tests
const success = runTests();
process.exit(success ? 0 : 1);
