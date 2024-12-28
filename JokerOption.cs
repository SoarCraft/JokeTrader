namespace JokeTrader;

using Bybit.Net.Enums;
using TorchSharp;

internal class JokerOption {
    public DateTime HistoryStart { get; } = new(2024, 11, 1, 0, 0, 0, DateTimeKind.Utc);

    public DateTime HistoryEnd { get; } = new(2024, 12, 1, 0, 0, 0, DateTimeKind.Utc);

    public Category Category => Category.Inverse;

    public string Symbol => "BTCUSDT";

    public KlineInterval KlineInterval => KlineInterval.FiveMinutes;

    public OpenInterestInterval InterestInterval => OpenInterestInterval.FiveMinutes;

    public DataPeriod Period => DataPeriod.FiveMinutes;

    public int BatchSize => 32;

    public int Epochs => 100;

    public int MaxWindow => 40;

    public int MinWindow => 10;

    public int NumHeads => 8;

    public int EmbedDim => 256;

    public int NumLayers => 6;

    public double Alpha => 0.5;

    public torch.Device Device { get; } = torch.cuda_is_available() ? torch.CUDA : torch.CPU;
}
