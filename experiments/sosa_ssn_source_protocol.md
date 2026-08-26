# SOSA and SSN source preparation and candidate pool protocol

## Scope

This stage locks raw source snapshots and enumerates which controlled fault
conditions can be constructed from each source unit. It does not select the
confirmatory cases. It does not fix the final sample size. It does not run an
extractor or a validator. It does not observe any model outcome.

The source unit is the experimental unit. Candidate rows are not independent
experimental observations. One source unit may be eligible for several fault
conditions. The later primary sample may assign at most one condition to
that source unit.

## Source families

### USGS daily water records

Four fixed queries to the modern USGS Water Data OGC API were captured for
2024-01-01: water temperature, precipitation, discharge, and gage height. Each
raw response and each parameter code metadata response is stored byte for byte
and protected by SHA-256. The preparation command is offline and never
fetches a URL again.

Only records with `approval_status = Approved` and no qualifier are eligible.
To reduce dependence among records from the same monitoring context, at most
one record is retained per `monitoring_location_id`. When a site occurs in more
than one snapshot, the parameter priority fixed in the source spec breaks the
tie before any model outcome exists.

The stable source unit identifier uses the time series identifier and
observation date. The API feature UUID is retained as provenance but is not
treated as a stable scientific identifier. The renderer states the provider,
date, parameter, statistic, value, unit, monitoring location, and time series
identifier. The clean graph builder added in a later stage must use the same
normalized record.

For candidate enumeration, a daily record can be represented as an observation
inside a one day observation collection. This record derived wrapper supports
the collection disjointness and member interval conditions. It does not claim
that the wrapper was present in the upstream JSON. The paper must call it an
adapter construction.

### Pinned W3C examples

Twelve byte identical Turtle files are copied from the pinned official W3C
SOSA and SSN repository. Each registry entry names one root token, a scenario
family, a deterministic English rendering, and only the fault conditions that
the example can support. The script checks both the file digest and presence of
the root token. The rendering is an adapter representation of the pinned RDF
example. It is not a quotation from a natural language corpus. It is not a
human gold annotation.

### EPA status

The EPA AQS website offers AirData files that require no credential. No
small immutable EPA snapshot is included in this version of the pool. EPA is
therefore marked deferred. It is not counted as an implemented source family.
It may be added in a later commit of the prepared state under the same raw
snapshot, hash, and offline preparation rules.

## Outputs and guards

`python -m src.prepare_sosa_ssn_sources` writes three deterministic artifacts:

- `experiments/sosa_ssn_source_units.jsonl`
- `experiments/sosa_ssn_candidate_pool.jsonl`
- `experiments/sosa_ssn_source_manifest.json`

The command fails if a raw digest changes, a USGS response has the wrong
parameter or date, a W3C root token is absent, a source unit ID is duplicated,
an unknown condition appears, or candidate coverage is insufficient for the
design target. The manifest reports accepted and rejected records, reasons,
source family and scenario family counts, condition eligibility counts, and
output digests.

Candidate eligibility is structural. It is not validator recall, repair
success, source faithfulness, or a model generated judgment.

## Required next stage

After this pool is reviewed, the next commit will define the sample size
rationale and deterministic allocation rule. It will select about 30 unique
source units per condition while balancing scenario families. Clean graph and
fault constructors, the project SHACL profile, prompts, runner, and analysis
plan follow. One audit of that complete prepared commit is required before any
confirmatory generation.
