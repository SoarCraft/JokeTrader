namespace JokeTrader;

using Bybit.Net.Enums;

internal class JokerOption {
    public DateTime HistoryStart { get; set; } = new(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    public DateTime HistoryEnd { get; set; } = new(2024, 12, 1, 0, 0, 0, DateTimeKind.Utc);

    public Category Category { get; set; } = Category.Inverse;

    public string[] Symbols { get; set; } = ["BTCUSDT"];
}
