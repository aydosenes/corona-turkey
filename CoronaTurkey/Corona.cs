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
        public float DailyTestCounts;

        [LoadColumn(3)]
        public float DailyDeaths;

    }
}
