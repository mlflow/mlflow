# Security Policy

MLflow and its community take security bugs seriously. We appreciate efforts to improve the security of MLflow
and follow the [GitHub coordinated disclosure of security vulnerabilities](https://docs.github.com/en/code-security/security-advisories/about-coordinated-disclosure-of-security-vulnerabilities#about-reporting-and-disclosing-vulnerabilities-in-projects-on-github)
for responsible disclosure and prompt mitigation. We are committed to working with security researchers to
resolve the vulnerabilities they discover.

## Supported Versions

The latest version of MLflow has continued support. If a critical vulnerability is found in the current version
of MLflow, we may opt to backport patches to previous versions.

## Reporting a Vulnerability

When finding a security vulnerability in Mlflow, [open an issue](https://github.com/mlflow/mlflow/issues/new?assignees=&labels=bug&template=bug_report_template.md&title=%5BBUG%5D%20Security%20Vulnerability
) on the Mlflow repo. Use `[BUG] Security Vulnerability` as title and *do not* mention vulnerability details in the issue.

An MLflow maintainer will:
  - Acknowledge the [bug](ISSUE_POLICY.md#bug-reports) during [triage](ISSUE_TRIAGE.rst)
  - Mark the issue as `priority/critical-urgent`
  - Open a draft [GitHub Security Advisory](https://docs.github.com/en/code-security/security-advisories/creating-a-security-advisory)
  to discuss the vulnerability details in private.

The private Security Advisory will be used to confirm the issue, prepare a fix, and publicly disclose it after the fix has been released.
