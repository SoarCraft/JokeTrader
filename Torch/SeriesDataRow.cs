namespace JokeTrader.Torch;

internal class SeriesDataRow {
    public DateTime Timestamp { get; set; }

    public double OpenPrice { get; set; }

    public double Volume { get; set; }

    public double BuyRatio { get; set; }

    public double SellRatio { get; set; }

    public double FundingRate { get; set; }

    public double OpenInterest { get; set; }

    public int Interval { get; set; }
}
