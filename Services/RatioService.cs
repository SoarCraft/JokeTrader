namespace JokeTrader.Services;

using System.Threading.Tasks;
using Bybit.Net.Interfaces.Clients;
using Microsoft.Extensions.Options;

internal class RatioService(IBybitRestClient restClient, JokerContext context, 
    IOptions<JokerOption> options, ILogger<RatioService> logger) : BackgroundService {
    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        var opt = options.Value;

        restClient.V5Api.ExchangeData.GetLongShortRatioAsync();

    }
}
