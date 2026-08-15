/*
7005SCN Individual Research Project
Original analytical view (V1)

Purpose:
- Preserved as development evidence.
- This version was later superseded after validation identified:
  * duplicate participant rows,
  * future session contamination,
  * GETDATE()-dependent outcomes,
  * target leakage from DaysSinceLastAttendance,
  * and invalid physiological values.

Do not use this view as the final modelling dataset.
*/

CREATE OR ALTER VIEW dbo.vw_MSc_ML_ParticipantRetentionDataset
AS
WITH ParticipantBase AS (
    SELECT
        p.ParticipantID,
        p.SaheliCardNumber,

        CONVERT(
            VARCHAR(64),
            HASHBYTES(
                'SHA2_256',
                CAST(p.ParticipantID AS VARCHAR(50))
            ),
            2
        ) AS ResearchParticipantID,

        CASE
            WHEN p.Age IS NOT NULL THEN p.Age
            WHEN p.DateOfBirth IS NOT NULL
                THEN DATEDIFF(YEAR, p.DateOfBirth, GETDATE())
            ELSE NULL
        END AS Age,

        CASE
            WHEN p.Age IS NULL AND p.DateOfBirth IS NULL THEN 'Unknown'
            WHEN COALESCE(
                p.Age,
                DATEDIFF(YEAR, p.DateOfBirth, GETDATE())
            ) < 18 THEN 'Under 18'
            WHEN COALESCE(
                p.Age,
                DATEDIFF(YEAR, p.DateOfBirth, GETDATE())
            ) BETWEEN 18 AND 29 THEN '18-29'
            WHEN COALESCE(
                p.Age,
                DATEDIFF(YEAR, p.DateOfBirth, GETDATE())
            ) BETWEEN 30 AND 44 THEN '30-44'
            WHEN COALESCE(
                p.Age,
                DATEDIFF(YEAR, p.DateOfBirth, GETDATE())
            ) BETWEEN 45 AND 59 THEN '45-59'
            WHEN COALESCE(
                p.Age,
                DATEDIFF(YEAR, p.DateOfBirth, GETDATE())
            ) >= 60 THEN '60+'
            ELSE 'Unknown'
        END AS AgeBand,

        p.Gender,
        p.Ethnicity,
        p.PreferredLanguage,
        p.Occupation,
        p.LivingAlone,
        p.CaringResponsibilities,
        p.ReferralReason,
        p.HeardAboutSaheli,
        p.Site,
        p.HasHealthConditionOrDisability,
        CAST(p.CreatedAt AS DATE) AS RegistrationDate
    FROM dbo.vw_Participants_Details p
),
AttendanceAgg AS (
    SELECT
        a.ParticipantId,
        COUNT(*) AS AttendanceRecords,
        SUM(CASE WHEN a.Attended = 1 THEN 1 ELSE 0 END) AS SessionsAttended,
        MIN(CAST(a.SessionDate AS DATE)) AS FirstAttendanceDate,
        MAX(CAST(a.SessionDate AS DATE)) AS LastAttendanceDate,
        COUNT(DISTINCT a.SessionId) AS UniqueSessionsBooked,
        COUNT(DISTINCT a.SessionName) AS UniqueActivityTypes,
        COUNT(DISTINCT a.SessionMonth) AS ActiveMonths,

        CASE
            WHEN COUNT(*) = 0 THEN NULL
            ELSE CAST(
                SUM(CASE WHEN a.Attended = 1 THEN 1 ELSE 0 END)
                AS FLOAT
            ) / COUNT(*)
        END AS AttendanceRate,

        DATEDIFF(
            DAY,
            MAX(CAST(a.SessionDate AS DATE)),
            GETDATE()
        ) AS DaysSinceLastAttendance

    FROM dbo.SessionAttendance a
    WHERE a.ParticipantId IS NOT NULL
    GROUP BY a.ParticipantId
),
AssessmentAgg AS (
    SELECT
        a.SaheliCardNumber,

        COUNT(*) AS AssessmentCount,
        MIN(CAST(a.AssessmentDate AS DATE)) AS FirstAssessmentDate,
        MAX(CAST(a.AssessmentDate AS DATE)) AS LastAssessmentDate,

        MAX(CASE WHEN a.AssessmentNumber = 1 THEN a.WeightKg END) AS InitialWeightKg,
        MAX(CASE WHEN a.AssessmentNumber = 1 THEN a.Bmivalue END) AS InitialBMI,
        MAX(CASE WHEN a.AssessmentNumber = 1 THEN a.HbA1c END) AS InitialHbA1c,
        MAX(CASE WHEN a.AssessmentNumber = 1 THEN a.HeartAge END) AS InitialHeartAge,
        MAX(CASE WHEN a.AssessmentNumber = 1 THEN a.ActiveDaysPerWeek END) AS InitialActiveDaysPerWeek,

        MAX(CASE WHEN a.AssessmentNumber = 1 THEN a.ConfidenceToJoin END) AS InitialConfidenceToJoin,
        MAX(CASE WHEN a.AssessmentNumber = 1 THEN a.FeelingOptimistic END) AS InitialFeelingOptimistic,
        MAX(CASE WHEN a.AssessmentNumber = 1 THEN a.FeelingUseful END) AS InitialFeelingUseful,
        MAX(CASE WHEN a.AssessmentNumber = 1 THEN a.FeelingRelaxed END) AS InitialFeelingRelaxed,
        MAX(CASE WHEN a.AssessmentNumber = 1 THEN a.FeelingConfident END) AS InitialFeelingConfident,
        MAX(CASE WHEN a.AssessmentNumber = 1 THEN a.FeelingCheerful END) AS InitialFeelingCheerful,

        MAX(a.AssessmentNumber) AS LatestAssessmentNumber,
        MAX(a.WeightKg) AS LatestWeightKg,
        MAX(a.Bmivalue) AS LatestBMI,
        MAX(a.HbA1c) AS LatestHbA1c,
        MAX(a.HeartAge) AS LatestHeartAge,
        MAX(a.ActiveDaysPerWeek) AS LatestActiveDaysPerWeek,
        MAX(a.ConfidenceToJoin) AS LatestConfidenceToJoin,

        AVG(CAST(a.LackCompanionship AS FLOAT)) AS AvgLackCompanionship,
        AVG(CAST(a.FeelLeftOut AS FLOAT)) AS AvgFeelLeftOut,
        AVG(CAST(a.FeelIsolated AS FLOAT)) AS AvgFeelIsolated,
        AVG(CAST(a.FeelingOptimistic AS FLOAT)) AS AvgFeelingOptimistic,
        AVG(CAST(a.FeelingUseful AS FLOAT)) AS AvgFeelingUseful,
        AVG(CAST(a.FeelingRelaxed AS FLOAT)) AS AvgFeelingRelaxed,
        AVG(CAST(a.FeelingConfident AS FLOAT)) AS AvgFeelingConfident,
        AVG(CAST(a.FeelingCheerful AS FLOAT)) AS AvgFeelingCheerful

    FROM dbo.vw_Participants_Assessments a
    GROUP BY a.SaheliCardNumber
),
FinalDataset AS (
    SELECT
        pb.ResearchParticipantID,
        pb.Age,
        pb.AgeBand,
        pb.Gender,
        pb.Ethnicity,
        pb.PreferredLanguage,
        pb.Occupation,
        pb.LivingAlone,
        pb.CaringResponsibilities,
        pb.ReferralReason,
        pb.HeardAboutSaheli,
        pb.Site,
        pb.HasHealthConditionOrDisability,
        pb.RegistrationDate,

        ISNULL(att.AttendanceRecords, 0) AS AttendanceRecords,
        ISNULL(att.SessionsAttended, 0) AS SessionsAttended,
        att.FirstAttendanceDate,
        att.LastAttendanceDate,
        ISNULL(att.UniqueSessionsBooked, 0) AS UniqueSessionsBooked,
        ISNULL(att.UniqueActivityTypes, 0) AS UniqueActivityTypes,
        ISNULL(att.ActiveMonths, 0) AS ActiveMonths,
        att.AttendanceRate,
        att.DaysSinceLastAttendance,

        ass.AssessmentCount,
        ass.FirstAssessmentDate,
        ass.LastAssessmentDate,
        ass.InitialWeightKg,
        ass.InitialBMI,
        ass.InitialHbA1c,
        ass.InitialHeartAge,
        ass.InitialActiveDaysPerWeek,
        ass.InitialConfidenceToJoin,
        ass.InitialFeelingOptimistic,
        ass.InitialFeelingUseful,
        ass.InitialFeelingRelaxed,
        ass.InitialFeelingConfident,
        ass.InitialFeelingCheerful,
        ass.LatestAssessmentNumber,
        ass.LatestWeightKg,
        ass.LatestBMI,
        ass.LatestHbA1c,
        ass.LatestHeartAge,
        ass.LatestActiveDaysPerWeek,
        ass.LatestConfidenceToJoin,
        ass.AvgLackCompanionship,
        ass.AvgFeelLeftOut,
        ass.AvgFeelIsolated,
        ass.AvgFeelingOptimistic,
        ass.AvgFeelingUseful,
        ass.AvgFeelingRelaxed,
        ass.AvgFeelingConfident,
        ass.AvgFeelingCheerful,

        DATEDIFF(
            DAY,
            pb.RegistrationDate,
            GETDATE()
        ) AS DaysSinceRegistration,

        CASE
            WHEN ISNULL(att.SessionsAttended, 0) = 0 THEN 0
            WHEN att.DaysSinceLastAttendance > 90 THEN 0
            ELSE 1
        END AS RetainedLabel,

        CASE
            WHEN ISNULL(att.SessionsAttended, 0) = 0 THEN 'Never Attended'
            WHEN att.DaysSinceLastAttendance > 90 THEN 'Disengaged'
            ELSE 'Retained'
        END AS RetentionStatus

    FROM ParticipantBase pb

    LEFT JOIN AttendanceAgg att
        ON pb.ParticipantID = att.ParticipantId

    LEFT JOIN AssessmentAgg ass
        ON pb.SaheliCardNumber = ass.SaheliCardNumber
)
SELECT *
FROM FinalDataset;
GO
