using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace CoronaTurkey
{
    public class DeathPrediction
    {
        [ColumnName("Score")]
        public float DailyDeathCounts;

    }
}
