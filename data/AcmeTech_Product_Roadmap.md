# AcmeTech Product Roadmap

This document describes planned and shipped features for the AcmeTech analytics platform.

## Current Release (v2.4)

The flagship product is Atlas, a real-time analytics dashboard. Key capabilities:

- Stream processing with sub-second latency for up to 50,000 events per second.
- Native support for PostgreSQL and MongoDB data sources.
- Role-based access control (RBAC) with three default roles: Viewer, Editor, Admin.
- Scheduled report delivery via email and Slack.

## Next Release (v2.5, target Q3)

- Anomaly detection engine using seasonal trend decomposition.
- Custom alert thresholds with per-workspace granularity.
- Data lineage view that traces every metric back to its source table.

## Longer term

- Mobile companion app for on-call teams.
- An embeddable dashboard SDK for external customers.
- SOC 2 Type II certification targeted for next year.

## Discontinued

- The legacy "Insight 1.x" pipeline was sunset in January. Customers using it are asked to
  migrate to Atlas before the end of the year.