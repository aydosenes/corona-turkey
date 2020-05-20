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
        public float TotalDeaths;

        [LoadColumn(3)]
        public float TotalRecovered;

        [LoadColumn(4)]
        public float ActiveCases;

        [LoadColumn(5)]
        public float DailyTestCases;

        [LoadColumn(6)]
        public float CaseIncreaseRate;

    }
}
