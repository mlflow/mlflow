# Security Policy

MLflow and its community take security bugs seriously. We appreciate efforts to improve the security of MLflow
and follow the [GitHub coordinated disclosure of security vulnerabilities](https://docs.github.com/en/code-security/security-advisories/about-coordinated-disclosure-of-security-vulnerabilities#about-reporting-and-disclosing-vulnerabilities-in-projects-on-github)
for responsible disclosure and prompt mitigation. We are committed to working with security researchers to
resolve the vulnerabilities they discover.

## Supported Versions

The latest version of MLflow has continued support. If a critical vulnerability is found in the current version
of MLflow, we may opt to backport patches to previous versions.

## Reporting a Vulnerability

When finding a security vulnerability in Mlflow, please perform the following actions:

- [Open an issue](https://github.com/mlflow/mlflow/issues/new?assignees=&labels=bug&template=bug_report_template.md&title=%5BBUG%5D%20Security%20Vulnerability) on the Mlflow repository. Ensure that you use `[BUG] Security Vulnerability` as the title and _do not_ mention any vulnerability details in the issue post.
- Send a notification [email](mailto:mlflow-oss-maintainers@googlegroups.com) to `mlflow-oss-maintainers@googlegroups.com` that contains, at a minimum:
  - The link to the filed issue stub.
  - Your GitHub handle.
  - Detailed information about the security vulnerability, evidence that supports the relevance of the finding and any reproducibility instructions for independent confirmation.

This first stage of reporting is to ensure that a rapid validation can occur without wasting the time and effort of a reporter. Future communication and vulnerability resolution will be conducted after validating
the veracity of the reported issue.

An MLflow maintainer will, after validating the report:

- Acknowledge the [bug](ISSUE_POLICY.md#bug-reports) during [triage](ISSUE_TRIAGE.rst)
- Mark the issue as `priority/critical-urgent`
- Open a draft [GitHub Security Advisory](https://docs.github.com/en/code-security/security-advisories/creating-a-security-advisory)
  to discuss the vulnerability details in private.

The private Security Advisory will be used to confirm the issue, prepare a fix, and publicly disclose it after the fix has been released.
