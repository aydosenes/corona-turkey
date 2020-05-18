using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CoronaTurkey
{
    class Program
    {
        static readonly string _trainDataPath = @"C:\Users\aydos\Desktop\repositories\Datasets\CoronaTurkey\covid_train.csv";
        static readonly string _testDataPath = @"C:\Users\aydos\Desktop\repositories\Datasets\CoronaTurkey\covid_test.csv";

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            TestSinglePrediction(mlContext, model);
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<Corona>(dataPath, hasHeader: true, separatorChar: ',');

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "DailyDeathCounts")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DayEncoded", inputColumnName: "Day"))
                .Append(mlContext.Transforms.Concatenate("Features", "DayEncoded", "DailyCaseCounts", "DailyTestCounts", "DailyDeathCounts"))
                .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataView);

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<Corona>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"Output : ");
            Console.WriteLine($"==========> R-Squared Score:{metrics.RSquared: %.###} ");
            Console.WriteLine($"==========> Root-Mean-Squared Error:{metrics.RootMeanSquaredError: #.###} ");
            Console.WriteLine();
            Console.WriteLine("Press Enter to Get Results...");
            Console.ReadLine();
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<Corona, DeathPrediction>(model);

            Console.WriteLine("Case Counts For Today : ");
            int caseCount = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine();

            Console.WriteLine("Test Counts For Today : ");
            var testCount = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine();

            Console.WriteLine("Actual Death Counts For Today : ");
            var actualNumber = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine();


            var coronaCaseSample = new Corona()
            {
                Day="Day 69",
                DailyCaseCounts= caseCount,
                DailyTestCounts= testCount,
                DailyDeathCounts = 0
            };

            var prediction = predictionFunction.Predict(coronaCaseSample);

            Console.WriteLine($"==========> Case Amount: {coronaCaseSample.DailyCaseCounts}");
            Console.WriteLine($"==========> Test Amount: {coronaCaseSample.DailyTestCounts}");
            Console.WriteLine($"==========> Actual Death Amount: {actualNumber}");
            Console.WriteLine($"==========> Predicted Death Amount: {prediction.DailyDeathCounts:0.####}");
            Console.ReadLine();
        }
    }
}

