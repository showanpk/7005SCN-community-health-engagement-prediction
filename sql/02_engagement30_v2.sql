/*
7005SCN Individual Research Project
Final research dataset: 30-day engagement prediction

Research snapshot date: 2026-08-08
Target:
  Engaged30Label = 1 if an eligible participant has an actual,
  non-cancelled attendance within 30 days of CRM registration.

Notes:
- One row per participant.
- Future sessions excluded.
- Only actual attended sessions are counted.
- Participants without a complete 30-day observation window are excluded.
- Records with attendance predating CRM registration are excluded because
  CreatedAt is not a reliable timeline anchor for those imported histories.
*/

CREATE OR ALTER VIEW dbo.vw_MSc_ML_Engagement30Dataset_V2
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
        p.HasHealthConditionOrDisability,
        CAST(p.CreatedAt AS date) AS RegistrationDate
    FROM dbo.vw_Participants_Details p
    WHERE p.ParticipantID IS NOT NULL
),
ActualAttendance AS
(
    SELECT
        a.ParticipantId,
        CAST(a.SessionDate AS date) AS SessionDate
    FROM dbo.vw_Report_AttendanceFlat a
    CROSS JOIN Params pr
    WHERE a.ParticipantId IS NOT NULL
      AND a.Attended = 1
      AND a.SessionDate IS NOT NULL
      AND CAST(a.SessionDate AS date) <= pr.AnalysisDate
      AND ISNULL(a.IsCancelled, 0) = 0
),
AttendanceSummary AS
(
    SELECT
        ParticipantId,
        MIN(SessionDate) AS FirstAttendanceDate
    FROM ActualAttendance
    GROUP BY ParticipantId
),
Prepared AS
(
    SELECT
        p.*,
        CASE
            WHEN p.DateOfBirth IS NOT NULL THEN
                DATEDIFF(YEAR, p.DateOfBirth, p.RegistrationDate)
                -
                CASE
                    WHEN DATEADD(
                        YEAR,
                        DATEDIFF(YEAR, p.DateOfBirth, p.RegistrationDate),
                        p.DateOfBirth
                    ) > p.RegistrationDate
                    THEN 1 ELSE 0
                END
            ELSE p.Age
        END AS AgeAtRegistration
    FROM ParticipantRaw p
)
SELECT
    CONVERT(
        varchar(64),
        HASHBYTES(
            'SHA2_256',
            CAST(p.ParticipantID AS varchar(50))
        ),
        2
    ) AS ResearchParticipantID,

    p.AgeAtRegistration AS Age,

    CASE
        WHEN p.AgeAtRegistration IS NULL THEN 'Unknown'
        WHEN p.AgeAtRegistration < 18 THEN 'Under 18'
        WHEN p.AgeAtRegistration BETWEEN 18 AND 29 THEN '18-29'
        WHEN p.AgeAtRegistration BETWEEN 30 AND 44 THEN '30-44'
        WHEN p.AgeAtRegistration BETWEEN 45 AND 59 THEN '45-59'
        ELSE '60+'
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
    p.RegistrationDate,

    CASE
        WHEN a.FirstAttendanceDate IS NOT NULL
         AND a.FirstAttendanceDate <= DATEADD(DAY, 30, p.RegistrationDate)
        THEN 1
        ELSE 0
    END AS Engaged30Label

FROM Prepared p

LEFT JOIN AttendanceSummary a
    ON p.ParticipantID = a.ParticipantId

CROSS JOIN Params pr

WHERE p.RegistrationDate IS NOT NULL
  AND p.RegistrationDate <= DATEADD(DAY, -30, pr.AnalysisDate)
  AND (
      a.FirstAttendanceDate IS NULL
      OR a.FirstAttendanceDate >= p.RegistrationDate
  );
GO
