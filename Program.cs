using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Taxi_Fare_Prediction_MLNet
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            //import or create training data
            IDataView trainingData = mlContext.Data.LoadFromTextFile<TaxiFare>("./Data/taxi-fare-train.csv", hasHeader: true, separatorChar: ',');

            

            //Specify data preparation and model training pipeline
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName:"Label",inputColumnName:"fare_amount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName:"VendorIdEncoded",inputColumnName:"vendor_id"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "rate_code"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PassengerCountEncoded", inputColumnName: "passenger_count"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TripDistanceEncoded", inputColumnName: "trip_distance"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "payment_type"))
                .Append(mlContext.Transforms.Concatenate("Features", new string[] { "VendorIdEncoded", "RateCodeEncoded", "PassengerCountEncoded", "TripDistanceEncoded", "PaymentTypeEncoded" }))
                .Append(mlContext.Regression.Trainers.Sdca());

            //Train model
            var model = pipeline.Fit(trainingData);

           //Evalluate
            var predictions = model.Transform(trainingData);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics output         ");
            Console.WriteLine($"*------------------------------------------------");

            Console.WriteLine($"*       R-Squared Score:      {metrics.RSquared:0.###}");

            Console.WriteLine($"*       Root-Mean-Squared Error:      {metrics.RootMeanSquaredError:#.###}");
            Console.WriteLine("Press Enter to continue...");
            

            //Make a prediction
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiFare, TaxiFarePrediction>(model);

            var newData = new TaxiFare
            {
                vendor_id = "CMT",
                rate_code = 1,
                passenger_count = 1,
                trip_distance = 3.8f,
                payment_type = "CRD"
            };
            var prediction = predictionFunction.Predict(newData);

            Console.WriteLine($"Predicted Fare - {prediction.predicted_fare_amount}, while actual fare: 17.5");
            Console.ReadLine();
        }
    }
}
