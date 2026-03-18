# Security Policy

MLflow and its community take security bugs seriously. We appreciate efforts to improve the security of MLflow
and follow the [GitHub coordinated disclosure of security vulnerabilities](https://docs.github.com/en/code-security/security-advisories/about-coordinated-disclosure-of-security-vulnerabilities#about-reporting-and-disclosing-vulnerabilities-in-projects-on-github)
for responsible disclosure and prompt mitigation. We are committed to working with security researchers to
resolve the vulnerabilities they discover.

## Supported Versions

The latest version of MLflow has continued support. If a critical vulnerability is found in the current version
of MLflow, we may opt to backport patches to previous versions.

## Bounty Platform Policy

Due to an excessive volume of low-quality, AI-generated reports (including duplicates, low-effort submissions, non-reproducible claims, and spam), **we no longer accept vulnerability reports through bounty platforms such as Huntr**. All reports submitted through such platforms will be automatically closed without review.

Please use the reporting process described below instead.

## Reporting a Vulnerability

When finding a security vulnerability in MLflow, please contact us at [mlflow-oss-maintainers@googlegroups.com](mailto:mlflow-oss-maintainers@googlegroups.com) with the following information:

- A clear description of the vulnerability
- Steps to reproduce the issue
- The potential impact of the vulnerability
- Your GitHub handle

An MLflow maintainer will, after validating the report:

- Acknowledge the [bug](ISSUE_POLICY.md#bug-reports) during [triage](ISSUE_TRIAGE.rst)
- Mark the issue as `priority/critical-urgent`
- Open a draft [GitHub Security Advisory](https://docs.github.com/en/code-security/security-advisories/creating-a-security-advisory)
  to discuss the vulnerability details in private.

The private Security Advisory will be used to confirm the issue, prepare a fix, and publicly disclose it after the fix has been released.
