namespace JokeTrader;

using Bybit.Net.Enums;

internal class JokerOption {
    public DateTime HistoryStart { get; set; } = new(2024, 11, 1, 0, 0, 0, DateTimeKind.Utc);

    public DateTime HistoryEnd { get; set; } = new(2024, 12, 1, 0, 0, 0, DateTimeKind.Utc);

    public Category Category { get; set; } = Category.Inverse;

    public string[] Symbols { get; set; } = ["BTCUSDT"];

    public KlineInterval KlineInterval { get; set; } = KlineInterval.FiveMinutes;

    public OpenInterestInterval InterestInterval { get; set; } = OpenInterestInterval.FiveMinutes;

    public DataPeriod Period { get; set; } = DataPeriod.FiveMinutes;

    public int BatchSize { get; set; } = 32;

    public int Epochs { get; set; } = 100;

    public int MaxWindow { get; set; } = 40;

    public int MinWindow { get; set; } = 10;
}
