# Data Privacy and Repository Scope

This repository deliberately excludes participant-level research data.

## Data not included

The following are not committed:

- names;
- addresses;
- telephone numbers;
- email addresses;
- CRM participant identifiers;
- row-level research extracts;
- model-ready participant-level CSV files;
- Azure SQL credentials or connection strings;
- production application configuration.

The research datasets include demographic and health-related variables and must therefore continue to be treated as protected research data even when direct identifiers are removed.

## Identifier handling

The analytical SQL creates a deterministic SHA-256 research identifier from the internal participant identifier. This should be treated as **pseudonymisation**, not as a guarantee of irreversible anonymisation.

## Repository contents

Only the following are included:

- SQL definitions;
- Python analysis code;
- aggregate model evaluation tables;
- feature-importance summaries;
- non-participant-level figures;
- methodological documentation.

## Reproduction

A reviewer with authorised access to the underlying research environment can regenerate the participant-level datasets by running the SQL scripts. The participant-level exports should remain within the approved secure research environment and should not be pushed to GitHub.
