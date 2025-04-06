namespace JokeTrader;

using System.ComponentModel.DataAnnotations;

public abstract class Concurrency { 
    [Timestamp]
    public byte[] Version { get; set; }
}

public class KLine : Concurrency {
    [Key]
    public DateTime StartTime { get; set; }

    public double OpenPrice { get; set; }

    public double HighPrice { get; set; }

    public double LowPrice { get; set; }

    public double ClosePrice { get; set; }

    public double Volume { get; set; }

    public Symbol Symbol { get; set; }
}

public class Ratio : Concurrency {
    [Key]
    public DateTime Timestamp { get; set; }

    public double BuyRatio { get; set; }

    public double SellRatio { get; set; }

    public Symbol Symbol { get; set; }
}

public class FundRate : Concurrency {
    [Key]
    public DateTime Timestamp { get; set; }

    public double FundingRate { get; set; }

    public Symbol Symbol { get; set; }
}

public class Interest : Concurrency {
    [Key]
    public DateTime Timestamp { get; set; }

    public double OpenInterest { get; set; }

    public Symbol Symbol { get; set; }
}

public class Symbol : Concurrency {
    [Key]
    public string Name { get; set; }

    public double MaxPrice { get; set; }

    public double MinPrice { get; set; }

    public double MinLeverage { get; set; }

    public double MaxLeverage { get; set; }

    public DateTime LastUpdated { get; set; }

    public List<KLine> KLines { get; set; } = [];

    public List<Ratio> Ratios { get; set; } = [];

    public List<FundRate> FundRates { get; set; } = [];

    public List<Interest> Interests { get; set; } = [];
}
