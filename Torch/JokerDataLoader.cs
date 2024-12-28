namespace JokeTrader.Torch;

using Bybit.Net.Enums;
using Microsoft.Extensions.Options;
using TorchSharp;

internal class JokerDataLoader(JokerContext context, IOptions<JokerOption> options, ILogger<JokerDataLoader> logger,
    ILogger<JokerDataEnumerator> enumLogger) : IAsyncEnumerable<(torch.Tensor, torch.Tensor)> {

    public JokerOption Opt => options.Value;

    public static int[] ViewSizes { get; } = new[] {
        KlineInterval.FiveMinutes,
        KlineInterval.FifteenMinutes,
        KlineInterval.OneHour,
        KlineInterval.FourHours,
        KlineInterval.OneDay
    }.Cast<int>().Select(x => x / 60).ToArray();

    public IAsyncEnumerator<(torch.Tensor, torch.Tensor)> GetAsyncEnumerator(CancellationToken cancellationToken = new()) {
        var randomViewSize = ViewSizes[Random.Shared.Next(ViewSizes.Length)];
        var randomWindowSize = Random.Shared.Next(this.Opt.MinWindow, this.Opt.MaxWindow + 1);

        logger.LogInformation($"View size: {randomViewSize}, Window size: {randomWindowSize}");
        return new JokerDataEnumerator(context, this.Opt, 1440, 35, enumLogger);
    }
}
