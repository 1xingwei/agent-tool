# AcmeTech Security & Compliance Guidelines

This guide consolidates the security controls that apply to all AcmeTech infrastructure.

## Access control

- Every employee is issued a single sign-on (SSO) account. Local passwords are not allowed.
- Multi-factor authentication is mandatory for all accounts with admin privileges.
- Access is reviewed quarterly. Accounts inactive for 90 days are automatically disabled.

## Data handling

- Credit card data is never stored on internal systems; payments go through a PCI Level 1 processor.
- Customer data at rest is encrypted with AES-256. Keys are rotated every 180 days.
- Backups are taken daily and tested for restore at least once per quarter.

## Incident response

- Suspected breaches must be reported to security@acmetech.com within one hour.
- The on-call rotation covers 24x7 coverage across three regions: NA, EMEA, APAC.
- Post-incident reviews produce a written report within two weeks.

## Acceptable use

- Personal use of company email is permitted in moderation.
- Downloading unapproved third-party tools to managed laptops is prohibited.