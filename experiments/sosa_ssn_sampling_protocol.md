# SOSA and SSN confirmatory sampling protocol

## Locked sample size

The confirmatory extension uses 30 independent source units for each of six
controlled fault conditions, for 180 source units in total. A source unit is
assigned to exactly one condition in the primary sample. Candidate rows for
different conditions are not counted as independent cases.

The 30 case cell is three times the preliminary ten case condition cell. One
case corresponds to 3.33 percentage points. At the most variable binomial
proportion, a conventional 95% Wilson interval at `n = 30` has a half width of
approximately 0.169. These quantities justify descriptive resolution and
resampling sensitivity analyses. They are not a population representative
power calculation. They do not license claims of statistical dominance.

## Allocation

The sample contains 168 USGS source units and all 12 pinned W3C example source
units. Each condition has exactly 30 cases. Four USGS scenario families are
balanced within each condition: water temperature, precipitation, stream
discharge, and gage height. Their totals over the full sample differ by at most
two cases.

The W3C examples are assigned by semantic suitability fixed before model
outcomes. The two genuine time bearing observation collections are assigned to
the temporal condition. The planned pH observation collection supplies the
W3C disjointness case. Observation, actuation, sampling, building energy,
seismic, and wind examples are distributed across the other conditions.

Within every USGS condition by scenario quota, eligible and unused source units
are ranked by SHA-256 over a fixed seed, condition, scenario family, and
source unit identifier. The lowest ranks are selected. This is deterministic.
It does not depend on input row order. It uses no experimental outcome.

## Independence and provenance guards

- Exactly 180 distinct source unit identifiers must be selected.
- USGS records must represent 168 distinct monitoring locations.
- No source unit may appear in more than one condition.
- Every selected pair must already occur in the locked candidate pool.
- The three source pool inputs are protected by SHA-256 and by the source pool
  commit identifier.
- Movie and Music preliminary cases are not eligible.
- Extraction, validation, grounding, and repair outcomes are prohibited
  selection inputs.

The selected sample remains controlled record derived data. It is not a
human annotated natural language benchmark. It is not a population sample of
all sensor observations.

## Status after selection

Selection locks the source units and their condition assignments. It does not
authorize model execution. The next stages construct clean graphs and the six
fault transformations, implement the project SHACL profile, and then lock
prompts, runner, and analysis. One audit of the complete prepared commit is
required before any confirmatory generation.
