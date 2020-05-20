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
                .Append(mlContext.Transforms.Concatenate("Features", "DateEncoded", "TotalDeaths", "TotalRecovered", "ActiveCases", 
                        "DailyTestCases", "CaseIncreaseRate"))
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

            Console.WriteLine("Date Number : ");
            var date = Convert.ToDateTime(Console.ReadLine()).ToString();
            Console.WriteLine();

            Console.WriteLine("Test Counts For Today : ");
            var testCount = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine();

            Console.WriteLine("Actual Case Counts For Today : ");
            var actualNumber = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine();


            var coronaCaseSample = new Corona()
            {
                Date = date,
                DailyCases = 0,
                TotalDeaths = 4199,
                TotalRecovered = 112895,
                ActiveCases = 34521,
                DailyTestCases = testCount,
                CaseIncreaseRate = 0.68f
            };

            var prediction = predictionFunction.Predict(coronaCaseSample);

            Console.WriteLine($"==========> Test Amount: {coronaCaseSample.DailyTestCases}");
            Console.WriteLine($"==========> Actual Case Amount: {actualNumber}");
            Console.WriteLine($"==========> Predicted Case Amount: {prediction.DailyCases:0.####}");
            Console.ReadLine();

        }

    }
}

