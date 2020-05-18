using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace CoronaTurkey
{
    public class Corona
    {
        [LoadColumn(0)]
        public string Day;

        [LoadColumn(1)]
        public float DailyCaseCounts;

        [LoadColumn(2)]
        public float DailyTestCounts;

        [LoadColumn(3)]
        public float DailyDeathCounts;

    }
}
