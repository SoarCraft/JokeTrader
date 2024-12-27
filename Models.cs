namespace JokeTrader;

using System.ComponentModel.DataAnnotations;

public abstract class Concurrency { 
    [Timestamp]
    public byte[] Version { get; set; }
}

public abstract class BasicKLine : Concurrency {
    [Key]
    public DateTime StartTime { get; set; }

    public double OpenPrice { get; set; }

    public double Volume { get; set; }

    public Symbol Symbol { get; set; }
}

public class BTCKLine : BasicKLine;

public abstract class BasicRatio : Concurrency {
    [Key]
    public DateTime Timestamp { get; set; }

    public double BuyRatio { get; set; }

    public double SellRatio { get; set; }

    public Symbol Symbol { get; set; }
}

public class BTCRatio : BasicRatio;

public abstract class BasicFundRate : Concurrency {
    [Key]
    public DateTime Timestamp { get; set; }

    public double FundingRate { get; set; }

    public Symbol Symbol { get; set; }
}

public class BTCFundRate : BasicFundRate;

public class BasicInterest : Concurrency {
    [Key]
    public DateTime Timestamp { get; set; }

    public double OpenInterest { get; set; }

    public Symbol Symbol { get; set; }
}

public class BTCInterest : BasicInterest;

public class Normalization {
    public string Feature { get; set; }

    public string SymbolId { get; set; }

    public Symbol Symbol { get; set; }

    public double Mean { get; set; }

    public double Std { get; set; }
}

public class Symbol : Concurrency {
    [Key]
    public string Name { get; set; }

    public double MaxPrice { get; set; }

    public double MinPrice { get; set; }

    public double MinLeverage { get; set; }

    public double MaxLeverage { get; set; }

    public DateTime LastUpdated { get; set; }

    public List<BTCKLine> BTCKLines { get; set; } = [];

    public List<BTCRatio> BTCRatios { get; set; } = [];

    public List<BTCFundRate> BTCFundRates { get; set; } = [];

    public List<BTCInterest> BTCInterests { get; set; } = [];

    public List<Normalization> Normalizations { get; set; } = [];
}
