namespace JokeTrader.Torch;

internal class SeriesFeatures {
    public double OpenPrice { get; set; }

    public double HighPrice { get; set; }

    public double LowPrice { get; set; }

    public double ClosePrice { get; set; }

    public double Volume { get; set; }

    public double BuyRatio { get; set; }

    public double SellRatio { get; set; }

    public double FundingRate { get; set; }

    public double OpenInterest { get; set; }

    public int Interval { get; set; }

    public double[] ToArray() => [
        this.OpenPrice,
        this.HighPrice,
        this.LowPrice,
        this.ClosePrice,
        this.Volume,
        this.BuyRatio,
        this.SellRatio,
        this.FundingRate,
        this.OpenInterest,
        this.Interval
    ];
}

internal class SeriesDataRow : SeriesFeatures {
    public DateTime Timestamp { get; set; }
}
