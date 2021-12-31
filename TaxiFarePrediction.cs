using Microsoft.ML.Data;

namespace Taxi_Fare_Prediction_MLNet
{
    internal class TaxiFarePrediction
    {
        [ColumnName("Score")]
        public float predicted_fare_amount { get; set; }
    }
}