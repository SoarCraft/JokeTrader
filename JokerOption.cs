namespace JokeTrader;

using Bybit.Net.Enums;
using TorchSharp;

internal class JokerOption {
    public DateTime HistoryStart { get; } = new(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    public DateTime HistoryEnd { get; } = new(2024, 12, 1, 0, 0, 0, DateTimeKind.Utc);

    public Category Category => Category.Inverse;

    public string Symbol => "BTCUSDT";

    public KlineInterval KlineInterval => KlineInterval.FiveMinutes;

    public OpenInterestInterval InterestInterval => OpenInterestInterval.FiveMinutes;

    public DataPeriod Period => DataPeriod.FiveMinutes;

    public int BatchSize => 100;

    public int Epochs => 200;

    public int MaxWindow => 40;

    public int MinWindow => 10;

    public int NumHeads => 16;

    public int EmbedDim => 1024;

    public int NumLayers => 24;

    public double Alpha => 0.7;

    public int Patience => 10;

    public torch.Device Device { get; } = torch.cuda_is_available() ? torch.CUDA : torch.CPU;
}
