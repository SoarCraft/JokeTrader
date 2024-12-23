namespace JokeTrader;

using System.ComponentModel.DataAnnotations;

public abstract class BasicKLine {
    [Key]
    public DateTime StartTime { get; set; }

    public double OpenPrice { get; set; }

    public Symbol Symbol { get; set; }
}

public class BTCKLine : BasicKLine;

public class Symbol {
    [Key]
    public string Name { get; set; }

    public double MaxOrderValue { get; set; }

    public double MinOrderValue { get; set; }

    public DateTime LastUpdated { get; set; }

    public List<BTCKLine> BTCKLines { get; set; } = [];
}
