namespace JokeTrader.Torch;

using System.Collections;
using Bybit.Net.Enums;
using Microsoft.Extensions.Options;
using TorchSharp;

internal class JokerDataLoader(IOptions<JokerOption> options, ILogger<JokerDataLoader> logger) : IEnumerable<(torch.Tensor, torch.Tensor)> {
    public JokerOption Opt => options.Value;

    public int[] ViewSizes { get; } = new[] {
        KlineInterval.FiveMinutes,
        KlineInterval.FifteenMinutes,
        KlineInterval.OneHour,
        KlineInterval.FourHours,
        KlineInterval.OneDay
    }.Cast<int>().Select(x => x / 60).ToArray();

    public IEnumerator<(torch.Tensor, torch.Tensor)> GetEnumerator() {
        var randomViewSize = this.ViewSizes[Random.Shared.Next(this.ViewSizes.Length)];
        var randomWindowSize = Random.Shared.Next(10, 40);
    }

    IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();
}
