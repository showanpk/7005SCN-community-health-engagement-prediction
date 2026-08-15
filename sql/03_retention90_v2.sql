/*
7005SCN Individual Research Project
Final research dataset: 90-day retention prediction

Research snapshot date: 2026-08-08

Observation window:
- first attendance through day 30.

Outcome window:
- days 31 through 90 after first attendance.

Target:
- Retained90Label = 1 if there is at least one actual, non-cancelled
  attendance during days 31-90.

Notes:
- One row per participant.
- Only participants with a complete 90-day follow-up window are included.
- Future sessions are excluded.
- Baseline assessment values are restricted to information available by
  the end of the first 30-day observation period.
- Implausible physiological values are treated as NULL in the research view.
*/

CREATE OR ALTER VIEW dbo.vw_MSc_ML_Retention90Dataset_V2
AS
WITH Params AS
(
    SELECT CAST('2026-08-08' AS date) AS AnalysisDate
),
ParticipantRaw AS
(
    SELECT DISTINCT
        p.ParticipantID,
        p.SaheliCardNumber,
        p.DateOfBirth,
        p.Age,
        p.Gender,
        p.Ethnicity,
        p.PreferredLanguage,
        p.Occupation,
        p.LivingAlone,
        p.CaringResponsibilities,
        p.ReferralReason,
        p.HeardAboutSaheli,
        p.Site,
        p.HasHealthConditionOrDisability
    FROM dbo.vw_Participants_Details p
    WHERE p.ParticipantID IS NOT NULL
),
ActualAttendance AS
(
    SELECT
        a.ParticipantId,
        a.SessionId,
        CAST(a.SessionDate AS date) AS SessionDate,
        NULLIF(LTRIM(RTRIM(a.ActivityName)), '') AS ActivityName
    FROM dbo.vw_Report_AttendanceFlat a
    CROSS JOIN Params pr
    WHERE a.ParticipantId IS NOT NULL
      AND a.Attended = 1
      AND a.SessionDate IS NOT NULL
      AND CAST(a.SessionDate AS date) <= pr.AnalysisDate
      AND ISNULL(a.IsCancelled, 0) = 0
),
FirstAttendance AS
(
    SELECT
        ParticipantId,
        MIN(SessionDate) AS FirstAttendanceDate
    FROM ActualAttendance
    GROUP BY ParticipantId
),
EligibleRaw AS
(
    SELECT
        p.*,
        f.FirstAttendanceDate,

        CASE
            WHEN p.DateOfBirth IS NOT NULL THEN
                DATEDIFF(YEAR, p.DateOfBirth, f.FirstAttendanceDate)
                -
                CASE
                    WHEN DATEADD(
                        YEAR,
                        DATEDIFF(YEAR, p.DateOfBirth, f.FirstAttendanceDate),
                        p.DateOfBirth
                    ) > f.FirstAttendanceDate
                    THEN 1 ELSE 0
                END
            ELSE p.Age
        END AS AgeAtFirstAttendance

    FROM ParticipantRaw p

    INNER JOIN FirstAttendance f
        ON p.ParticipantID = f.ParticipantId

    CROSS JOIN Params pr

    WHERE f.FirstAttendanceDate <= DATEADD(DAY, -90, pr.AnalysisDate)
),
EarlyAttendance AS
(
    SELECT
        e.ParticipantID,

        COUNT(*) AS EarlySessions30,

        COUNT(DISTINCT a.SessionId)
            AS EarlyUniqueSessions30,

        COUNT(DISTINCT a.ActivityName)
            AS EarlyUniqueActivityTypes30,

        COUNT(
            DISTINCT
            DATEDIFF(
                DAY,
                e.FirstAttendanceDate,
                a.SessionDate
            ) / 7
        ) AS EarlyActiveWeeks30

    FROM EligibleRaw e

    INNER JOIN ActualAttendance a
        ON e.ParticipantID = a.ParticipantId
       AND a.SessionDate >= e.FirstAttendanceDate
       AND a.SessionDate <= DATEADD(
            DAY,
            30,
            e.FirstAttendanceDate
       )

    GROUP BY e.ParticipantID
),
RetentionOutcome AS
(
    SELECT
        e.ParticipantID,

        CASE
            WHEN COUNT(a.SessionId) > 0 THEN 1
            ELSE 0
        END AS Retained90Label

    FROM EligibleRaw e

    LEFT JOIN ActualAttendance a
        ON e.ParticipantID = a.ParticipantId
       AND a.SessionDate > DATEADD(
            DAY,
            30,
            e.FirstAttendanceDate
       )
       AND a.SessionDate <= DATEADD(
            DAY,
            90,
            e.FirstAttendanceDate
       )

    GROUP BY e.ParticipantID
),
AssessmentSource AS
(
    SELECT DISTINCT
        a.SaheliCardNumber,
        CAST(a.AssessmentDate AS date) AS AssessmentDate,
        a.AssessmentNumber,
        a.WeightKg,
        a.Bmivalue,
        a.HbA1c,
        a.HeartAge,
        a.ActiveDaysPerWeek,
        a.ConfidenceToJoin,
        a.FeelingOptimistic,
        a.FeelingUseful,
        a.FeelingRelaxed,
        a.FeelingConfident,
        a.FeelingCheerful
    FROM dbo.vw_Participants_Assessments a
    WHERE a.SaheliCardNumber IS NOT NULL
),
AssessmentCandidates AS
(
    SELECT
        e.ParticipantID,
        a.AssessmentDate,
        a.AssessmentNumber,
        a.WeightKg,
        a.Bmivalue,
        a.HbA1c,
        a.HeartAge,
        a.ActiveDaysPerWeek,
        a.ConfidenceToJoin,
        a.FeelingOptimistic,
        a.FeelingUseful,
        a.FeelingRelaxed,
        a.FeelingConfident,
        a.FeelingCheerful,

        ROW_NUMBER() OVER
        (
            PARTITION BY e.ParticipantID
            ORDER BY
                a.AssessmentDate,
                a.AssessmentNumber
        ) AS rn

    FROM EligibleRaw e

    INNER JOIN AssessmentSource a
        ON e.SaheliCardNumber = a.SaheliCardNumber

    WHERE a.AssessmentDate <= DATEADD(
        DAY,
        30,
        e.FirstAttendanceDate
    )
),
BaselineAssessment AS
(
    SELECT
        ParticipantID,

        AssessmentDate AS BaselineAssessmentDate,

        CASE
            WHEN WeightKg BETWEEN 25 AND 300
            THEN CAST(WeightKg AS float)
        END AS BaselineWeightKg,

        CASE
            WHEN Bmivalue BETWEEN 10 AND 80
            THEN CAST(Bmivalue AS float)
        END AS BaselineBMI,

        CASE
            WHEN HbA1c BETWEEN 20 AND 200
            THEN CAST(HbA1c AS float)
        END AS BaselineHbA1c,

        CASE
            WHEN HeartAge BETWEEN 18 AND 120
            THEN HeartAge
        END AS BaselineHeartAge,

        CASE
            WHEN ActiveDaysPerWeek BETWEEN 0 AND 7
            THEN ActiveDaysPerWeek
        END AS BaselineActiveDaysPerWeek,

        CASE
            WHEN ConfidenceToJoin BETWEEN 0 AND 10
            THEN ConfidenceToJoin
        END AS BaselineConfidenceToJoin,

        CASE
            WHEN FeelingOptimistic BETWEEN 1 AND 5
            THEN FeelingOptimistic
        END AS BaselineFeelingOptimistic,

        CASE
            WHEN FeelingUseful BETWEEN 1 AND 5
            THEN FeelingUseful
        END AS BaselineFeelingUseful,

        CASE
            WHEN FeelingRelaxed BETWEEN 1 AND 5
            THEN FeelingRelaxed
        END AS BaselineFeelingRelaxed,

        CASE
            WHEN FeelingConfident BETWEEN 1 AND 5
            THEN FeelingConfident
        END AS BaselineFeelingConfident,

        CASE
            WHEN FeelingCheerful BETWEEN 1 AND 5
            THEN FeelingCheerful
        END AS BaselineFeelingCheerful

    FROM AssessmentCandidates
    WHERE rn = 1
)
SELECT
    CONVERT(
        varchar(64),
        HASHBYTES(
            'SHA2_256',
            CAST(e.ParticipantID AS varchar(50))
        ),
        2
    ) AS ResearchParticipantID,

    e.AgeAtFirstAttendance AS Age,

    CASE
        WHEN e.AgeAtFirstAttendance IS NULL THEN 'Unknown'
        WHEN e.AgeAtFirstAttendance < 18 THEN 'Under 18'
        WHEN e.AgeAtFirstAttendance BETWEEN 18 AND 29 THEN '18-29'
        WHEN e.AgeAtFirstAttendance BETWEEN 30 AND 44 THEN '30-44'
        WHEN e.AgeAtFirstAttendance BETWEEN 45 AND 59 THEN '45-59'
        ELSE '60+'
    END AS AgeBand,

    e.Gender,
    e.Ethnicity,
    e.PreferredLanguage,
    e.Occupation,
    e.LivingAlone,
    e.CaringResponsibilities,
    e.ReferralReason,
    e.HeardAboutSaheli,
    e.Site,
    e.HasHealthConditionOrDisability,

    e.FirstAttendanceDate,

    ISNULL(x.EarlySessions30, 0)
        AS EarlySessions30,

    ISNULL(x.EarlyUniqueSessions30, 0)
        AS EarlyUniqueSessions30,

    ISNULL(x.EarlyUniqueActivityTypes30, 0)
        AS EarlyUniqueActivityTypes30,

    ISNULL(x.EarlyActiveWeeks30, 0)
        AS EarlyActiveWeeks30,

    b.BaselineAssessmentDate,
    b.BaselineWeightKg,
    b.BaselineBMI,
    b.BaselineHbA1c,
    b.BaselineHeartAge,
    b.BaselineActiveDaysPerWeek,
    b.BaselineConfidenceToJoin,
    b.BaselineFeelingOptimistic,
    b.BaselineFeelingUseful,
    b.BaselineFeelingRelaxed,
    b.BaselineFeelingConfident,
    b.BaselineFeelingCheerful,

    r.Retained90Label,

    CASE
        WHEN r.Retained90Label = 1
        THEN 'Retained'
        ELSE 'Not Retained'
    END AS Retention90Status

FROM EligibleRaw e

LEFT JOIN EarlyAttendance x
    ON e.ParticipantID = x.ParticipantID

LEFT JOIN BaselineAssessment b
    ON e.ParticipantID = b.ParticipantID

INNER JOIN RetentionOutcome r
    ON e.ParticipantID = r.ParticipantID;
GO
