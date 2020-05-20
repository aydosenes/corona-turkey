using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace CoronaTurkey
{
    class Program
    {
        //static readonly string dataPath = @"C:\Users\aydos\Desktop\repositories\Datasets\Covid19-Turkey.csv";
        static string basePath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        static string dataPath = Path.Combine(basePath, "Covid19-Turkey.csv");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            IDataView dataView = mlContext.Data.LoadFromTextFile<Corona>(dataPath, hasHeader: true, separatorChar: ',');
            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1);
            IDataView trainData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "DailyCases")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DateEncoded", inputColumnName: "Date"))
                .Append(mlContext.Transforms.Concatenate("Features", "DateEncoded", "DailyDeaths","DailyTestCases", "TotalDeaths"))
                .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(trainData);

            var predictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine("********************| CORONAVIRUS IN TURKEY |********************");
            Console.WriteLine();
            Console.WriteLine($"Output : ");
            Console.WriteLine($"==========> R-Squared Score:{metrics.RSquared: %.###} ");
            Console.WriteLine($"==========> Root-Mean-Squared Error:{metrics.RootMeanSquaredError: #.###} ");
            Console.WriteLine();
            Console.WriteLine("Press Enter to Get Results...");
            Console.ReadLine();

            var predictionFunction = mlContext.Model.CreatePredictionEngine<Corona, CasePrediction>(model);

            Console.WriteLine("Date : ");
            var date = Convert.ToDateTime(Console.ReadLine()).ToString();
            Console.WriteLine();
            Console.WriteLine("How many people died today? : ");
            var dailyDeaths = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine();
            Console.WriteLine("How many tests applied today? : ");
            var testCount = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine();
            Console.WriteLine("How many people died at total? : ");
            var totalDeaths = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine();
            Console.WriteLine("The number trying to be predicted : ");
            var actualNumber = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine();


            var coronaCaseSample = new Corona()
            {
                Date = date,
                DailyCases = 0,
                DailyDeaths = dailyDeaths,
                DailyTestCases = testCount,
                TotalDeaths = totalDeaths,

            };

            var prediction = predictionFunction.Predict(coronaCaseSample);

            Console.WriteLine($"==========> Test Amount: {coronaCaseSample.DailyTestCases}");
            Console.WriteLine($"==========> Actual Case Amount: {actualNumber}");
            Console.WriteLine($"==========> Predicted Case Amount: {prediction.DailyCases:0.####}");
            Console.ReadLine();

        }

    }
}

