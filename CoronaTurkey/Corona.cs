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
        public float DailyDeaths;

        [LoadColumn(3)]
        public float DailyTestCases;

        [LoadColumn(4)]
        public float TotalDeaths;

    }
}
