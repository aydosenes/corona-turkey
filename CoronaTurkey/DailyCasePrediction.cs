using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace CoronaTurkey
{
    public class DailyCasePrediction
    {
        [ColumnName("Score")]
        public float DailyCases;

    }
}
