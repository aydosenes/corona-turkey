using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace CoronaTurkey
{
    public class CasePrediction
    {
        [ColumnName("Score")]
        public float DailyCases;

    }
}
