# Requirements Document: Problem-Driven Optimization (性价比方案)

## Introduction

This document specifies a **lightweight problem-driven optimization** that addresses three critical mechanisms from the MCM Problem C statement with minimal implementation effort (1-2 hours total).

Based on careful analysis of the problem statement, we focus on three high-impact optimizations:

1. **Within-Week Relative Standardization**: Address score inflation across 34 seasons (30 min)
2. **Teflon Index Feature**: Capture the "Jerry Rice phenomenon" (30 min)
3. **Elimination Accuracy Metric**: Align evaluation with problem objective (30 min)

Current baseline: Judge R² 81.73%, Fan R² 75.48% (17 features, 2777 records)
Target: Add 2-3 problem-driven features and 1 new metric with minimal code changes.

## Glossary

- **Within-Week Standardization**: Normalizing scores relative to same-week competitors (addresses score inflation)
- **Teflon Index**: Measure of contestant's ability to survive despite low judge scores (Jerry Rice phenomenon)
- **Elimination Accuracy**: Metric measuring correct prediction of eliminated contestant (problem-aligned evaluation)

## Requirements

### Requirement 1: Within-Week Relative Standardization (30 min)

**User Story:** As a data scientist, I want to normalize scores relative to same-week competitors, so that I can account for score inflation across 34 seasons.

#### Acceptance Criteria

1. THE system SHALL compute within-week mean and standard deviation for each Season-Week combination
2. THE system SHALL create feature `Judge_Score_Rel_Week` = (Score - Week_Mean) / Week_Std
3. WHEN Week_Std is zero, THE system SHALL set `Judge_Score_Rel_Week` to 0
4. THE system SHALL add this feature to the existing 17-feature dataset
5. THE system SHALL retrain models and report performance change

**Implementation**: Add ~10 lines of code to feature engineering pipeline.

### Requirement 2: Teflon Index Feature (30 min)

**User Story:** As a data scientist, I want to quantify the "Jerry Rice phenomenon" (low judge scores but high survival), so that I can model contestants with strong fan bases.

#### Acceptance Criteria

1. THE system SHALL compute judge ranking within each week (1 = highest score)
2. THE system SHALL compute survival ranking based on total survival weeks
3. THE system SHALL create `Judge_Fan_Divergence` = Judge_Rank - Survival_Rank
4. THE system SHALL create `Teflon_Index` = cumulative sum of positive divergences
5. THE system SHALL add these 2 features to the dataset
6. THE system SHALL verify Jerry Rice (Season 2) has high Teflon_Index

**Implementation**: Add ~15 lines of code to feature engineering pipeline.

### Requirement 3: Elimination Accuracy Metric (30 min)

**User Story:** As a data scientist, I want to evaluate models based on elimination prediction accuracy, so that I align evaluation with the problem objective: "determine which couple to eliminate."

#### Acceptance Criteria

1. THE system SHALL compute `Elimination_Accuracy` = % of weeks where predicted eliminated contestant matches actual
2. THE system SHALL compute `Bottom_3_Accuracy` = % of weeks where actual eliminated is in predicted bottom 3
3. THE system SHALL report these metrics alongside R² scores
4. THE system SHALL generate a simple comparison table

**Implementation**: Add ~20 lines of code to evaluation module.
