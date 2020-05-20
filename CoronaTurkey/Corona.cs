using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace CoronaTurkey
{
    public class Corona
    {
        [LoadColumn(0)]
        public string Date;

        [LoadColumn(1)]
        public float DailyCases;

        [LoadColumn(2)]
        public float TotalCases;

        [LoadColumn(3)]
        public float TotalDeaths;

        [LoadColumn(4)]
        public float TotalRecovered;

        [LoadColumn(5)]
        public float ActiveCases;

        [LoadColumn(6)]
        public float DailyTestCases;

        [LoadColumn(7)]
        public float TotalIntensiveCare;

        [LoadColumn(8)]
        public float IntubatedCases;

        [LoadColumn(9)]
        public float CaseIncreaseRate;

        [LoadColumn(10)]
        public float DailyCaseTestRate;

        [LoadColumn(11)]
        public float RecoveredActiveCaseRate;

        [LoadColumn(12)]
        public float DeathActiveCaseRate;

        [LoadColumn(13)]
        public float ActiveCasePopulationRate;

    }
}
